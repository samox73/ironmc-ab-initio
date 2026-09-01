use nalgebra::Vector3;
use num_complex::Complex64;

use crate::{diagram::Diagram, model::Model};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightRatio {
    pub modulus: f64,
    pub phase: Complex64,
}

/// One arc's vertex convention. Equal endpoint momenta reduce this to `|g(k,q)|²`.
pub fn arc_factor(
    model: &dyn Model,
    k_emit: Vector3<f64>,
    k_absorb: Vector3<f64>,
    q: Vector3<f64>,
    branch: usize,
) -> Complex64 {
    model.vertex(k_emit, q, branch) * model.vertex(k_absorb, q, branch).conj()
}

pub fn weight_modulus(diagram: &Diagram, model: &dyn Model) -> f64 {
    log_modulus_and_phase(diagram, model).0.exp()
}

pub fn phase(diagram: &Diagram, model: &dyn Model) -> Complex64 {
    log_modulus_and_phase(diagram, model).1
}

pub fn add_phonon_ratio(before: &Diagram, after: &Diagram, model: &dyn Model) -> WeightRatio {
    ratio(before, after, model)
}

pub fn remove_phonon_ratio(before: &Diagram, after: &Diagram, model: &dyn Model) -> WeightRatio {
    ratio(before, after, model)
}

pub fn change_tau_ratio(before: &Diagram, after: &Diagram, model: &dyn Model) -> WeightRatio {
    ratio(before, after, model)
}

pub fn change_internal_tau_ratio(
    before: &Diagram,
    after: &Diagram,
    model: &dyn Model,
) -> WeightRatio {
    ratio(before, after, model)
}

fn ratio(before: &Diagram, after: &Diagram, model: &dyn Model) -> WeightRatio {
    let (before_log, before_phase) = log_modulus_and_phase(before, model);
    let (after_log, after_phase) = log_modulus_and_phase(after, model);
    let log_ratio = after_log - before_log;
    WeightRatio {
        modulus: if log_ratio > f64::MAX.ln() { f64::INFINITY } else { log_ratio.exp() },
        phase: after_phase * before_phase.conj(),
    }
}

fn log_modulus_and_phase(diagram: &Diagram, model: &dyn Model) -> (f64, Complex64) {
    let mut log_modulus = 0.0;
    let mut total_phase = Complex64::new(1.0, 0.0);

    for key in 0..diagram.tail {
        let next = key + 1;
        let dt = diagram.v(next).tau - diagram.v(key).tau;
        log_modulus -= (model.electron_energy(diagram.v(key).p_out) - diagram.mu) * dt;
    }

    for (left, right) in diagram.arc_keys() {
        let vertex = diagram.v(left);
        let factor = arc_factor(
            model,
            diagram.v(left).p_out,
            diagram.incoming_momentum(right),
            vertex.q,
            vertex.branch,
        );
        let modulus = factor.norm();
        if modulus == 0.0 || !modulus.is_finite() {
            return (f64::NEG_INFINITY, Complex64::new(1.0, 0.0));
        }
        let phonon_energy = model.phonon_energy(vertex.q, vertex.branch);
        debug_assert!(
            phonon_energy >= model.phonon_frequency_cutoff(),
            "a nonzero vertex survived the phonon-frequency cutoff"
        );
        log_modulus += modulus.ln()
            - model.log_momentum_measure()
            - phonon_energy * (diagram.v(right).tau - vertex.tau);
        total_phase *= factor / modulus;
    }

    let norm = total_phase.norm();
    if norm > 0.0 {
        total_phase /= norm;
    }
    (log_modulus, total_phase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FrohlichModel;

    #[test]
    fn single_arc_convention_is_vertex_modulus_squared() {
        let model = FrohlichModel::new(1.0);
        let gamma = Vector3::zeros();
        let q = Vector3::new(0.4, -0.2, 0.1);
        let mut diagram = Diagram::new(-1.2, q, 4.0, 0, 4);
        let (left, right) = diagram.insert_arc(0.5, 1.5, q, 0);
        let factor = arc_factor(
            &model,
            diagram.v(left).p_out,
            diagram.incoming_momentum(right),
            q,
            0,
        );
        assert_eq!(diagram.v(left).p_out, gamma);
        assert_eq!(diagram.incoming_momentum(right), gamma);
        assert!((factor.re - model.vertex(gamma, q, 0).norm_sqr()).abs() < 1.0e-12);
        assert_eq!(factor.im, 0.0);
    }
}
