use nalgebra::Vector3;
use num_complex::Complex64;
use rmc_core::mc::Measurement;
use rmc_core::{Merge, Result, RmcError};
use rmc_stats::ScalarJackknife;
use serde::{Deserialize, Serialize};

use crate::diagram::Diagram;
use crate::model::Model;
use crate::weight;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Estimate {
    pub mean: f64,
    pub stderr: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GreenFunction {
    pub tau: Vec<f64>,
    pub value: Vec<Complex64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Report {
    pub e0: Estimate,
    pub z: Estimate,
    pub e0_thermo: Estimate,
    pub mean_order: Estimate,
    pub sign: Estimate,
    pub green: GreenFunction,
    pub green_energy: Vec<Estimate>,
}

#[derive(Clone, Debug, Default)]
struct Batch {
    hist: Vec<Complex64>,
    eloc_hist: Vec<Complex64>,
    zeroth: u64,
    phase_sum: Complex64,
    order_sum: f64,
    count: u64,
}

impl Batch {
    fn new(num_bins: usize) -> Self {
        Self {
            hist: vec![Complex64::new(0.0, 0.0); num_bins],
            eloc_hist: vec![Complex64::new(0.0, 0.0); num_bins],
            ..Self::default()
        }
    }

    fn add_assign(&mut self, other: &Self) {
        for (lhs, rhs) in self.hist.iter_mut().zip(&other.hist) {
            *lhs += rhs;
        }
        for (lhs, rhs) in self.eloc_hist.iter_mut().zip(&other.eloc_hist) {
            *lhs += rhs;
        }
        self.zeroth += other.zeroth;
        self.phase_sum += other.phase_sum;
        self.order_sum += other.order_sum;
        self.count += other.count;
    }
}

#[derive(Clone, Debug)]
pub struct MeasurementData {
    batches: Vec<Batch>,
}

impl Merge for MeasurementData {
    fn merge(mut self, other: Self) -> Self {
        self.batches.extend(other.batches);
        self
    }
}

pub(crate) struct DiagMeasurement {
    model: std::sync::Arc<dyn Model>,
    batches: Vec<Batch>,
    batch_len: usize,
    next: usize,
    max_tau: f64,
}

impl DiagMeasurement {
    pub(crate) fn new(
        model: std::sync::Arc<dyn Model>,
        num_bins: usize,
        n_batches: usize,
        expected_samples: usize,
        max_tau: f64,
    ) -> Self {
        Self {
            model,
            batches: (0..n_batches).map(|_| Batch::new(num_bins)).collect(),
            batch_len: expected_samples.max(1).div_ceil(n_batches).max(1),
            next: 0,
            max_tau,
        }
    }
}

impl Measurement<Diagram> for DiagMeasurement {
    type Output = MeasurementData;

    fn measure(&mut self, diagram: &Diagram) {
        let batch = (self.next / self.batch_len).min(self.batches.len() - 1);
        let phase = weight::phase(diagram, self.model.as_ref());
        let bin = ((diagram.tau() / self.max_tau * self.batches[batch].hist.len() as f64) as usize)
            .min(self.batches[batch].hist.len() - 1);
        let data = &mut self.batches[batch];
        data.hist[bin] += phase;
        data.eloc_hist[bin] += phase * local_energy(diagram, self.model.as_ref());
        data.zeroth += u64::from(diagram.order == 0);
        data.phase_sum += phase;
        data.order_sum += diagram.order as f64;
        data.count += 1;
        self.next += 1;
    }

    fn finish(self) -> Self::Output {
        MeasurementData {
            batches: self.batches,
        }
    }
}

pub(crate) fn summarize(
    data: MeasurementData,
    model: &dyn Model,
    mu: f64,
    p_ext: Vector3<f64>,
    max_tau: f64,
    tau_fit: f64,
    fixed_tau: bool,
) -> Result<Report> {
    let nonempty: Vec<_> = data
        .batches
        .into_iter()
        .filter(|batch| batch.count > 0)
        .collect();
    if nonempty.len() < 2 {
        return Err(RmcError::InvalidState(
            "decay jackknife needs at least two non-empty batches".to_string(),
        ));
    }
    let total = aggregate(&nonempty, None);
    let num_bins = total.hist.len();
    let bin_width = max_tau / num_bins as f64;
    let tau: Vec<_> = (0..num_bins)
        .map(|bin| (bin as f64 + 0.5) * bin_width)
        .collect();
    let (green_values, e0, z, e0_leave_one, z_leave_one) = if fixed_tau {
        (vec![], f64::NAN, f64::NAN, vec![], vec![])
    } else {
        let norm0 = zeroth_norm(model.electron_energy(p_ext) - mu, max_tau);
        let green_values = normalized_green(&total, norm0, bin_width)?;
        let (e0, z) = decay_fit(&tau, &green_values, mu, tau_fit)?;
        let mut e0_leave_one = Vec::with_capacity(nonempty.len());
        let mut z_leave_one = Vec::with_capacity(nonempty.len());
        for omitted in 0..nonempty.len() {
            let sample = aggregate(&nonempty, Some(omitted));
            let green = normalized_green(&sample, norm0, bin_width)?;
            let (sample_e0, sample_z) = decay_fit(&tau, &green, mu, tau_fit)?;
            e0_leave_one.push(sample_e0);
            z_leave_one.push(sample_z);
        }
        (green_values, e0, z, e0_leave_one, z_leave_one)
    };

    let green_energy = energy_curve(&total, &nonempty);
    let first_fit_bin = tau.partition_point(|&t| t < tau_fit);
    let e0_thermo_mean = energy_ratio(&total, first_fit_bin);
    let e0_thermo_leave_one: Vec<_> = (0..nonempty.len())
        .map(|omitted| energy_ratio(&aggregate(&nonempty, Some(omitted)), first_fit_bin))
        .filter(|value| value.is_finite())
        .collect();

    let sign_mean = total.phase_sum.re / total.count as f64;
    let order_mean = total.order_sum / total.count as f64;
    let sign_batches = nonempty
        .iter()
        .map(|batch| batch.phase_sum.re / batch.count as f64);
    let order_batches = nonempty
        .iter()
        .map(|batch| batch.order_sum / batch.count as f64);
    let sign_jackknife = ScalarJackknife::from_values(sign_batches);
    let order_jackknife = ScalarJackknife::from_values(order_batches);

    Ok(Report {
        e0: Estimate {
            mean: e0,
            stderr: nonlinear_jackknife_error(&e0_leave_one),
        },
        z: Estimate {
            mean: z,
            stderr: nonlinear_jackknife_error(&z_leave_one),
        },
        e0_thermo: Estimate {
            mean: e0_thermo_mean,
            stderr: nonlinear_jackknife_error(&e0_thermo_leave_one),
        },
        mean_order: Estimate {
            mean: order_mean,
            stderr: order_jackknife.standard_error().unwrap_or(0.0),
        },
        sign: Estimate {
            mean: sign_mean,
            stderr: sign_jackknife.standard_error().unwrap_or(0.0),
        },
        green: GreenFunction {
            tau: if fixed_tau { vec![] } else { tau },
            value: green_values,
        },
        green_energy,
    })
}

fn local_energy(diagram: &Diagram, model: &dyn Model) -> f64 {
    let tau = diagram.tau();
    let electron = (0..diagram.tail)
        .map(|key| {
            let dt = diagram.v(key + 1).tau - diagram.v(key).tau;
            model.electron_energy(diagram.v(key).p_out) * dt
        })
        .sum::<f64>();
    let phonon = diagram
        .arc_keys()
        .into_iter()
        .map(|(left, right)| {
            let vertex = diagram.v(left);
            model.phonon_energy(vertex.q, vertex.branch) * (diagram.v(right).tau - vertex.tau)
        })
        .sum::<f64>();
    (electron + phonon - 2.0 * diagram.order as f64) / tau
}

fn energy_ratio(batch: &Batch, first_bin: usize) -> f64 {
    let denominator = batch.hist[first_bin..]
        .iter()
        .map(|value| value.re)
        .sum::<f64>();
    let numerator = batch.eloc_hist[first_bin..]
        .iter()
        .map(|value| value.re)
        .sum::<f64>();
    if denominator == 0.0 {
        f64::NAN
    } else {
        numerator / denominator
    }
}

fn energy_curve(total: &Batch, batches: &[Batch]) -> Vec<Estimate> {
    (0..total.hist.len())
        .map(|bin| {
            let mean = if total.hist[bin].re == 0.0 {
                f64::NAN
            } else {
                total.eloc_hist[bin].re / total.hist[bin].re
            };
            let leave_one: Vec<_> = (0..batches.len())
                .filter_map(|omitted| {
                    let sample = aggregate(batches, Some(omitted));
                    (sample.hist[bin].re != 0.0)
                        .then(|| sample.eloc_hist[bin].re / sample.hist[bin].re)
                })
                .collect();
            Estimate {
                mean,
                stderr: nonlinear_jackknife_error(&leave_one),
            }
        })
        .collect()
}

fn aggregate(batches: &[Batch], omitted: Option<usize>) -> Batch {
    let mut total = Batch::new(batches[0].hist.len());
    for (index, batch) in batches.iter().enumerate() {
        if Some(index) != omitted {
            total.add_assign(batch);
        }
    }
    total
}

fn normalized_green(batch: &Batch, norm0: f64, bin_width: f64) -> Result<Vec<Complex64>> {
    if batch.zeroth == 0 {
        return Err(RmcError::InvalidState(
            "Green-function normalization has no order-0 samples".to_string(),
        ));
    }
    let scale = norm0 / (batch.zeroth as f64 * bin_width);
    Ok(batch.hist.iter().map(|value| value * scale).collect())
}

fn zeroth_norm(energy: f64, max_tau: f64) -> f64 {
    if energy.abs() < 1.0e-12 {
        max_tau
    } else {
        -(-energy * max_tau).exp_m1() / energy
    }
}

fn decay_fit(tau: &[f64], green: &[Complex64], mu: f64, tau_fit: f64) -> Result<(f64, f64)> {
    let points: Vec<_> = tau
        .iter()
        .zip(green)
        .filter_map(|(&x, value)| {
            let modulus = value.norm();
            (x >= tau_fit && modulus > 0.0 && modulus.is_finite()).then_some((x, modulus.ln()))
        })
        .collect();
    if points.len() < 2 {
        return Err(RmcError::InvalidState(
            "decay window contains fewer than two populated bins".to_string(),
        ));
    }
    let n = points.len() as f64;
    let mean_x = points.iter().map(|point| point.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|point| point.1).sum::<f64>() / n;
    let denominator = points
        .iter()
        .map(|point| (point.0 - mean_x).powi(2))
        .sum::<f64>();
    if denominator == 0.0 {
        return Err(RmcError::InvalidState(
            "decay fit has zero tau span".to_string(),
        ));
    }
    let slope = points
        .iter()
        .map(|point| (point.0 - mean_x) * (point.1 - mean_y))
        .sum::<f64>()
        / denominator;
    let intercept = mean_y - slope * mean_x;
    Ok((mu - slope, intercept.exp()))
}

fn nonlinear_jackknife_error(estimates: &[f64]) -> f64 {
    if estimates.len() < 2 {
        return f64::NAN;
    }
    let n = estimates.len() as f64;
    let mean = estimates.iter().sum::<f64>() / n;
    (((n - 1.0) / n)
        * estimates
            .iter()
            .map(|estimate| (estimate - mean).powi(2))
            .sum::<f64>())
    .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_fit_recovers_exponential() {
        let tau = [1.0, 2.0, 3.0, 4.0];
        let green: Vec<_> = tau
            .iter()
            .map(|&t| Complex64::new(0.7 * f64::exp(-1.2 * t), 0.0))
            .collect();
        let (e0, z) = decay_fit(&tau, &green, -2.0, 1.0).unwrap();
        assert!((e0 + 0.8).abs() < 1.0e-12);
        assert!((z - 0.7).abs() < 1.0e-12);
    }

    #[test]
    fn local_energy_uses_unshifted_segment_and_arc_energies() {
        use crate::model::FrohlichModel;

        let model = FrohlichModel::new(1.0);
        let q = Vector3::new(1.0, 0.0, 0.0);
        let mut diagram = Diagram::new(99.0, Vector3::zeros(), 4.0, 0, 4);
        diagram.set_vertex_tau(diagram.tail, 4.0);
        diagram.insert_arc(1.0, 3.0, q, 0);
        // Electron: 0·1 + (q²/2)·2 + 0·1 = 1; phonon: 1·2; vertices: -2.
        assert!((local_energy(&diagram, &model) - 0.25).abs() < 1.0e-12);
    }
}
