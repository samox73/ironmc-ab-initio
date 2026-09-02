use nalgebra::Vector3;
use num_complex::Complex64;

use crate::{diagram::Diagram, model::Model};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightRatio {
    pub modulus: f64,
    pub phase: Complex64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalRatio {
    pub log_modulus: f64,
    pub phase: Complex64,
}

impl LocalRatio {
    pub fn modulus(self) -> f64 {
        if self.log_modulus > f64::MAX.ln() {
            f64::INFINITY
        } else {
            self.log_modulus.exp()
        }
    }
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

pub fn insert_arc_ratio(
    diagram: &Diagram,
    model: &dyn Model,
    tau1: f64,
    tau2: f64,
    q: Vector3<f64>,
    branch: usize,
) -> LocalRatio {
    let changed_arcs = changed_arc_factors(diagram, model, tau1, tau2, q, None);
    let mut log_modulus =
        segment_change(diagram, model, tau1, tau2, |p| p - q) + changed_arcs.log_modulus;
    let first = segment_at(diagram, tau1);
    let last = segment_at(diagram, tau2);
    let factor = arc_factor(
        model,
        diagram.v(first).p_out - q,
        diagram.v(last).p_out - q,
        q,
        branch,
    );
    let modulus = factor.norm();
    if modulus == 0.0 || !modulus.is_finite() {
        return LocalRatio {
            log_modulus: f64::NEG_INFINITY,
            phase: Complex64::new(1.0, 0.0),
        };
    }
    log_modulus += modulus.ln()
        - model.log_momentum_measure()
        - model.phonon_energy(q, branch) * (tau2 - tau1);
    LocalRatio {
        log_modulus,
        phase: changed_arcs.phase * factor / modulus,
    }
}

pub fn remove_arc_ratio(
    diagram: &Diagram,
    model: &dyn Model,
    left: usize,
    right: usize,
) -> LocalRatio {
    assert_eq!(diagram.v(left).link, right);
    let vertex = diagram.v(left);
    let factor = arc_factor(
        model,
        vertex.p_out,
        diagram.incoming_momentum(right),
        vertex.q,
        vertex.branch,
    );
    let modulus = factor.norm();
    if modulus == 0.0 || !modulus.is_finite() {
        return LocalRatio {
            log_modulus: f64::INFINITY,
            phase: Complex64::new(1.0, 0.0),
        };
    }
    let changed_arcs = changed_arc_factors(
        diagram,
        model,
        vertex.tau,
        diagram.v(right).tau,
        -vertex.q,
        Some((left, right)),
    );
    let mut log_modulus = segment_change(diagram, model, vertex.tau, diagram.v(right).tau, |p| {
        p + vertex.q
    });
    log_modulus += changed_arcs.log_modulus - modulus.ln()
        + model.log_momentum_measure()
        + model.phonon_energy(vertex.q, vertex.branch) * (diagram.v(right).tau - vertex.tau);
    LocalRatio {
        log_modulus,
        phase: changed_arcs.phase * (factor / modulus).conj(),
    }
}

pub fn shift_vertex_tau_ratio(
    diagram: &Diagram,
    model: &dyn Model,
    key: usize,
    new_tau: f64,
) -> LocalRatio {
    assert!(key > diagram.head && key < diagram.tail);
    let delta = new_tau - diagram.v(key).tau;
    let before = model.electron_energy(diagram.v(key - 1).p_out);
    let after = model.electron_energy(diagram.v(key).p_out);
    let vertex = diagram.v(key);
    let phonon = model.phonon_energy(vertex.q, vertex.branch)
        * if vertex.link > key { delta } else { -delta };
    LocalRatio {
        log_modulus: (after - before) * delta + phonon,
        phase: Complex64::new(1.0, 0.0),
    }
}

pub fn shift_tail_ratio(diagram: &Diagram, model: &dyn Model, new_tau: f64) -> LocalRatio {
    let delta = new_tau - diagram.tau();
    LocalRatio {
        log_modulus: -(model.electron_energy(diagram.v(diagram.tail - 1).p_out) - diagram.mu)
            * delta,
        phase: Complex64::new(1.0, 0.0),
    }
}

fn segment_change(
    diagram: &Diagram,
    model: &dyn Model,
    tau1: f64,
    tau2: f64,
    changed: impl Fn(Vector3<f64>) -> Vector3<f64>,
) -> f64 {
    let mut key = segment_at(diagram, tau1);
    let mut out = 0.0;
    while key < diagram.tail && diagram.v(key).tau < tau2 {
        let left = tau1.max(diagram.v(key).tau);
        let right = tau2.min(diagram.v(key + 1).tau);
        if right > left {
            let p = diagram.v(key).p_out;
            out -= (model.electron_energy(changed(p)) - model.electron_energy(p)) * (right - left);
        }
        key += 1;
    }
    out
}

fn changed_arc_factors(
    diagram: &Diagram,
    model: &dyn Model,
    tau1: f64,
    tau2: f64,
    inserted_q: Vector3<f64>,
    skip: Option<(usize, usize)>,
) -> LocalRatio {
    let mut ratio = LocalRatio {
        log_modulus: 0.0,
        phase: Complex64::new(1.0, 0.0),
    };
    let start = diagram.vertices.partition_point(|vertex| vertex.tau < tau1);
    let end = diagram
        .vertices
        .partition_point(|vertex| vertex.tau <= tau2)
        .min(diagram.tail);
    for key in start..end {
        let linked = diagram.v(key).link;
        if linked == usize::MAX {
            continue;
        }
        let (left, right) = if key < linked { (key, linked) } else { (linked, key) };
        let old_emit = diagram.v(left).p_out;
        let old_absorb = diagram.incoming_momentum(right);
        let emit_changed = diagram.v(left).tau >= tau1 && diagram.v(left).tau < tau2;
        let absorb_changed = diagram.v(right).tau > tau1 && diagram.v(right).tau <= tau2;
        if skip == Some((left, right))
            || (!emit_changed && !absorb_changed)
            || (key == right && emit_changed)
        {
            continue;
        }
        let vertex = diagram.v(left);
        let old = arc_factor(model, old_emit, old_absorb, vertex.q, vertex.branch);
        let new = arc_factor(
            model,
            old_emit - inserted_q * f64::from(emit_changed),
            old_absorb - inserted_q * f64::from(absorb_changed),
            vertex.q,
            vertex.branch,
        );
        let old_modulus = old.norm();
        let new_modulus = new.norm();
        if old_modulus == 0.0 || new_modulus == 0.0 {
            ratio.log_modulus = if new_modulus == 0.0 { f64::NEG_INFINITY } else { f64::INFINITY };
            ratio.phase = Complex64::new(1.0, 0.0);
            return ratio;
        }
        ratio.log_modulus += new_modulus.ln() - old_modulus.ln();
        ratio.phase *= new / new_modulus * (old / old_modulus).conj();
    }
    ratio
}

pub(crate) fn segment_at(diagram: &Diagram, tau: f64) -> usize {
    diagram
        .vertices
        .partition_point(|vertex| vertex.tau <= tau)
        .saturating_sub(1)
        .min(diagram.tail - 1)
}

pub fn sync_cache(diagram: &mut Diagram, model: &dyn Model) {
    let (log_modulus, phase) = log_modulus_and_phase(diagram, model);
    diagram.set_cached_weight(log_modulus, phase);
}

pub fn log_modulus_and_phase(diagram: &Diagram, model: &dyn Model) -> (f64, Complex64) {
    let mut log_modulus = 0.0;
    let mut total_phase = Complex64::new(1.0, 0.0);

    for (p_out, dt) in diagram.segments() {
        log_modulus -= (model.electron_energy(p_out) - diagram.mu) * dt;
    }

    for (left, right) in diagram.arcs() {
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
    use crate::model::{FrohlichModel, Model};
    use rand::{Rng, RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    struct ComplexModel;

    impl Model for ComplexModel {
        fn n_branches(&self) -> usize {
            1
        }

        fn electron_energy(&self, k: Vector3<f64>) -> f64 {
            k.norm_squared() / 2.0
        }

        fn phonon_energy(&self, q: Vector3<f64>, _nu: usize) -> f64 {
            0.8 + 0.1 * q[0].abs()
        }

        fn vertex(&self, k: Vector3<f64>, q: Vector3<f64>, _nu: usize) -> Complex64 {
            Complex64::from_polar(1.0 + 0.1 * q.norm_squared(), k[0] + q[1])
        }

        fn propose_phonon(&self, _rng: &mut dyn RngCore) -> (Vector3<f64>, usize, f64) {
            (Vector3::new(0.2, 0.1, -0.1), 0, 1.0)
        }

        fn propose_phonon_pdf(&self, _q: Vector3<f64>, nu: usize) -> f64 {
            f64::from(nu == 0)
        }
    }

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

    #[test]
    fn local_ratios_match_from_scratch_on_random_diagrams() {
        let frohlich = FrohlichModel::new(1.0);
        let complex = ComplexModel;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xc1c2);
        for model in [&frohlich as &dyn Model, &complex] {
            for _ in 0..100 {
                let mut diagram = Diagram::new(-1.1, Vector3::new(0.1, -0.05, 0.02), 12.0, 0, 24);
                for _ in 0..rng.gen_range(0..=20) {
                    let mut tau = [rng.gen_range(0.01..5.99), rng.gen_range(0.01..5.99)];
                    tau.sort_by(f64::total_cmp);
                    if tau[1] - tau[0] < 1.0e-12 {
                        continue;
                    }
                    let q = Vector3::new(
                        rng.gen_range(-0.8..0.8),
                        rng.gen_range(-0.8..0.8),
                        rng.gen_range(-0.8..0.8),
                    );
                    diagram.insert_arc(tau[0], tau[1], q, 0);
                }

                let tau1 = rng.gen_range(0.01..5.0);
                let tau2 = rng.gen_range(tau1 + 0.001..5.99);
                let q = Vector3::new(
                    rng.gen_range(-0.8..0.8),
                    rng.gen_range(-0.8..0.8),
                    rng.gen_range(-0.8..0.8),
                );
                let local = insert_arc_ratio(&diagram, model, tau1, tau2, q, 0);
                let mut after = diagram.clone();
                after.insert_arc(tau1, tau2, q, 0);
                assert_matches_scratch(&diagram, &after, model, local);

                if diagram.order > 0 {
                    let (left, right) = diagram.nth_arc(rng.gen_range(0..diagram.order)).unwrap();
                    let local = remove_arc_ratio(&diagram, model, left, right);
                    let mut after = diagram.clone();
                    after.remove_arc(left, right);
                    assert_matches_scratch(&diagram, &after, model, local);

                    let key = rng.gen_range(1..diagram.tail);
                    let new_tau = diagram.v(key - 1).tau
                        + 0.37 * (diagram.v(key + 1).tau - diagram.v(key - 1).tau);
                    let local = shift_vertex_tau_ratio(&diagram, model, key, new_tau);
                    let mut after = diagram.clone();
                    after.set_vertex_tau(key, new_tau);
                    assert_matches_scratch(&diagram, &after, model, local);
                }

                let lower = diagram.v(diagram.tail - 1).tau;
                let new_tau = lower + 0.61 * (diagram.max_tau - lower);
                let local = shift_tail_ratio(&diagram, model, new_tau);
                let mut after = diagram.clone();
                after.set_vertex_tau(after.tail, new_tau);
                assert_matches_scratch(&diagram, &after, model, local);
            }
        }
    }

    fn assert_matches_scratch(
        before: &Diagram,
        after: &Diagram,
        model: &dyn Model,
        local: LocalRatio,
    ) {
        let (before_log, before_phase) = log_modulus_and_phase(before, model);
        let (after_log, after_phase) = log_modulus_and_phase(after, model);
        assert!(
            (local.log_modulus - (after_log - before_log)).abs() < 1.0e-10,
            "{} != {}",
            local.log_modulus,
            after_log - before_log
        );
        assert!((local.phase - after_phase * before_phase.conj()).norm() < 1.0e-10);
    }
}
