use num_complex::Complex64;
use rmc_core::mc::Measurement;
use rmc_core::{Merge, Result, RmcError};
use rmc_stats::ScalarJackknife;
use serde::{Deserialize, Serialize};

use crate::diagram::Diagram;
use crate::model::Model;

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
    /// Decay-fit estimators and the G(τ) curve exist only in free-τ mode; a
    /// fixed-τ run samples at τ = max_tau exactly, so they are `None` there
    /// (omitted from JSON) and `e0_thermo` is the estimator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub e0: Option<Estimate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z: Option<Estimate>,
    pub e0_thermo: Estimate,
    pub mean_order: Estimate,
    pub sign: Estimate,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub green: Option<GreenFunction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub green_energy: Option<Vec<Estimate>>,
    /// Total MC steps completed across all chains. Below `chains · max_steps` the
    /// run was truncated by a wall-clock deadline and the sample count is no
    /// longer reproducible from the seed alone.
    #[serde(default)]
    pub steps_done: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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

    /// `total − self`: the leave-one-out replicate in O(bins) instead of a
    /// fresh O(n_batches · bins) aggregation — the jackknife was quadratic in
    /// the batch count, a wall at GPU-scale chain populations.
    fn subtracted_from(&self, total: &Self) -> Self {
        let mut out = total.clone();
        for (lhs, rhs) in out.hist.iter_mut().zip(&self.hist) {
            *lhs -= rhs;
        }
        for (lhs, rhs) in out.eloc_hist.iter_mut().zip(&self.eloc_hist) {
            *lhs -= rhs;
        }
        out.zeroth -= self.zeroth;
        out.phase_sum -= self.phase_sum;
        out.order_sum -= self.order_sum;
        out.count -= self.count;
        out
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeasurementData {
    batches: Vec<Batch>,
}

impl Merge for MeasurementData {
    fn merge(mut self, other: Self) -> Self {
        self.batches.extend(other.batches);
        self
    }
}

pub struct DiagMeasurement {
    model: std::sync::Arc<dyn Model>,
    batches: Vec<Batch>,
    batch_len: usize,
    next: usize,
    max_tau: f64,
}

impl DiagMeasurement {
    pub fn new(
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

impl DiagMeasurement {
    /// Record one per-cycle sample from its raw scalars — the entry point of
    /// the flat/GPU drivers' host-side measurement replay (their per-cycle
    /// sample log carries exactly these fields).
    pub fn measure_sample(&mut self, tau: f64, order: usize, phase: Complex64, local_energy: f64) {
        let batch = (self.next / self.batch_len).min(self.batches.len() - 1);
        let bin = ((tau / self.max_tau * self.batches[batch].hist.len() as f64) as usize)
            .min(self.batches[batch].hist.len() - 1);
        let data = &mut self.batches[batch];
        data.hist[bin] += phase;
        data.eloc_hist[bin] += phase * local_energy;
        data.zeroth += u64::from(order == 0);
        data.phase_sum += phase;
        data.order_sum += order as f64;
        data.count += 1;
        self.next += 1;
    }

    /// `finish` as a plain method (the trait consumes `self`; drivers that do
    /// not run through the `Runner` call this directly).
    pub fn into_output(self) -> MeasurementData {
        Measurement::<Diagram>::finish(self)
    }
}

impl Measurement<Diagram> for DiagMeasurement {
    type Output = MeasurementData;

    fn measure(&mut self, diagram: &Diagram) {
        let local_energy = local_energy(diagram, self.model.as_ref());
        self.measure_sample(
            diagram.tau(),
            diagram.order,
            diagram.cached_phase(),
            local_energy,
        );
    }

    fn finish(mut self) -> Self::Output {
        // A deadline-truncated run leaves a partial trailing batch; below half
        // a batch its noisy mean would distort the equal-weight jackknives, so
        // fold it into its predecessor (still contiguous in MC time). A
        // complete run's last batch is short by at most n_batches − 1 samples
        // (batch_len rounds up) and never triggers this.
        if let Some(last) = self.batches.iter().rposition(|batch| batch.count > 0) {
            if last > 0 && self.batches[last].count.saturating_mul(2) < self.batch_len as u64 {
                let tail = std::mem::take(&mut self.batches[last]);
                self.batches[last - 1].add_assign(&tail);
            }
        }
        MeasurementData {
            batches: self.batches,
        }
    }
}

/// Reduce merged batch data to the final report. Takes only the one model
/// scalar it needs (`ε(p_ext)`, the Green-function normalization energy) so
/// offline merges never reload a vertex table.
pub fn summarize(
    data: MeasurementData,
    electron_energy_pext: f64,
    mu: f64,
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
    let total = aggregate(&nonempty);
    let num_bins = total.hist.len();
    let bin_width = max_tau / num_bins as f64;
    let tau: Vec<_> = (0..num_bins)
        .map(|bin| (bin as f64 + 0.5) * bin_width)
        .collect();
    let (green_values, e0, z, e0_leave_one, z_leave_one) = if fixed_tau {
        (vec![], f64::NAN, f64::NAN, vec![], vec![])
    } else {
        let norm0 = zeroth_norm(electron_energy_pext - mu, max_tau);
        let green_values = normalized_green(&total, norm0, bin_width)?;
        let (e0, z) = decay_fit(&tau, &green_values, mu, tau_fit)?;
        // A degenerate replicate (e.g. all order-0 samples in the omitted
        // batch) must not abort the run — the total above already normalized.
        // Skip it, but say so, and never jackknife from fewer than two.
        let mut e0_leave_one = Vec::with_capacity(nonempty.len());
        let mut z_leave_one = Vec::with_capacity(nonempty.len());
        let mut skipped = 0_usize;
        for omitted in 0..nonempty.len() {
            let sample = nonempty[omitted].subtracted_from(&total);
            match normalized_green(&sample, norm0, bin_width)
                .and_then(|green| decay_fit(&tau, &green, mu, tau_fit))
            {
                Ok((sample_e0, sample_z)) => {
                    e0_leave_one.push(sample_e0);
                    z_leave_one.push(sample_z);
                }
                Err(_) => skipped += 1,
            }
        }
        if skipped > 0 {
            log::warn!(
                "decay-fit jackknife skipped {skipped}/{} degenerate replicates; \
                 the reported e0/z stderr uses only the survivors",
                nonempty.len()
            );
        }
        if e0_leave_one.len() < 2 {
            return Err(RmcError::InvalidState(
                "decay-fit jackknife has fewer than two valid replicates".to_string(),
            ));
        }
        (green_values, e0, z, e0_leave_one, z_leave_one)
    };

    let green_energy = energy_curve(&total, &nonempty);
    let first_fit_bin = tau.partition_point(|&t| t < tau_fit);
    let e0_thermo_mean = energy_ratio(&total, first_fit_bin);
    if !e0_thermo_mean.is_finite() {
        if fixed_tau {
            // In fixed-τ mode this is the only estimator — fail loudly instead
            // of reporting NaN.
            return Err(RmcError::InvalidState(
                "thermodynamic estimator has no samples in the fit window".to_string(),
            ));
        }
        log::warn!("thermodynamic estimator has no samples past tau_fit; e0_thermo is NaN");
    }
    let all_thermo: Vec<_> = nonempty
        .iter()
        .map(|batch| energy_ratio(&batch.subtracted_from(&total), first_fit_bin))
        .collect();
    let e0_thermo_leave_one: Vec<_> = all_thermo
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if e0_thermo_leave_one.len() < all_thermo.len() {
        log::warn!(
            "thermodynamic jackknife dropped {}/{} non-finite replicates; \
             the reported e0_thermo stderr uses only the survivors",
            all_thermo.len() - e0_thermo_leave_one.len(),
            all_thermo.len()
        );
    }
    if fixed_tau && e0_thermo_leave_one.len() < 2 {
        return Err(RmcError::InvalidState(
            "thermodynamic jackknife has fewer than two valid replicates".to_string(),
        ));
    }

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
        e0: (!fixed_tau).then(|| Estimate {
            mean: e0,
            stderr: nonlinear_jackknife_error(&e0_leave_one),
        }),
        z: (!fixed_tau).then(|| Estimate {
            mean: z,
            stderr: nonlinear_jackknife_error(&z_leave_one),
        }),
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
        green: (!fixed_tau).then(|| GreenFunction {
            tau,
            value: green_values,
        }),
        green_energy: (!fixed_tau).then_some(green_energy),
        steps_done: 0,
    })
}

/// Thermodynamic estimator E_loc = (1/τ)[Σ_s ε·Δτ + Σ_a ω·Δτ − 2·order],
/// walking the diagram through the same `segments()`/`arcs()` iterators the
/// weight uses, so both stay on one convention.
fn local_energy(diagram: &Diagram, model: &dyn Model) -> f64 {
    let electron = diagram
        .segments()
        .map(|(p_out, dt)| model.electron_energy(p_out) * dt)
        .sum::<f64>();
    let phonon = diagram
        .arcs()
        .map(|(left, right)| {
            let vertex = diagram.v(left);
            model.phonon_energy(vertex.q, vertex.branch) * (diagram.v(right).tau - vertex.tau)
        })
        .sum::<f64>();
    (electron + phonon - 2.0 * diagram.order as f64) / diagram.tau()
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
            let leave_one: Vec<_> = batches
                .iter()
                .filter_map(|batch| {
                    let hist = total.hist[bin].re - batch.hist[bin].re;
                    let eloc = total.eloc_hist[bin].re - batch.eloc_hist[bin].re;
                    (hist != 0.0).then(|| eloc / hist)
                })
                .collect();
            Estimate {
                mean,
                stderr: nonlinear_jackknife_error(&leave_one),
            }
        })
        .collect()
}

fn aggregate(batches: &[Batch]) -> Batch {
    let mut total = Batch::new(batches[0].hist.len());
    for batch in batches {
        total.add_assign(batch);
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
    use nalgebra::Vector3;

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

    #[test]
    fn finish_folds_small_trailing_batch_into_predecessor() {
        use crate::model::FrohlichModel;

        let model = std::sync::Arc::new(FrohlichModel::new(1.0));
        let mut diagram = Diagram::new(0.0, Vector3::zeros(), 4.0, 0, 4);
        diagram.set_vertex_tau(diagram.tail, 1.0);
        let counts = |samples: usize| {
            let mut measurement = DiagMeasurement::new(model.clone(), 4, 10, 100, 4.0);
            for _ in 0..samples {
                measurement.measure(&diagram);
            }
            measurement
                .finish()
                .batches
                .iter()
                .map(|batch| batch.count)
                .filter(|&count| count > 0)
                .collect::<Vec<_>>()
        };
        // Truncated: batch_len = 10, the 2-sample tail (< half a batch) folds
        // into its predecessor.
        assert_eq!(counts(32), vec![10, 10, 12]);
        // A tail of at least half a batch stays its own block.
        assert_eq!(counts(35), vec![10, 10, 10, 5]);
        // Complete run: untouched.
        assert_eq!(counts(100), vec![10; 10]);
    }
}
