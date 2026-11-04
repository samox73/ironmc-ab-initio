//! The ab-initio update set (plan B1/B3): reuse the Fröhlich engine's five
//! vertex-agnostic moves unchanged and reimplement only the three that change an
//! arc's momentum `q`, multiplying their acceptance by the material reweight
//! `R(q)` (see [`crate::vertex`]). Frohlich is imported, never modified.
//!
//! For a **radial** `R(|q|)`, `ChangeQDirection` (fixed `|q|`) and `ChangeTopology`
//! (permutes the arc-`q` multiset) leave `∏R` invariant, so only Add / Remove /
//! ChangeQModulus carry the factor — a rigorous target reweight of the validated
//! Fröhlich sampler. At `R ≡ 1`/const the three collapse to the Fröhlich moves
//! bit-for-bit (identical RNG stream, guarded by `tests/frohlich_parity.rs`).

use nalgebra::Vector3;
use rand::{Rng, RngExt};
use rmc_core::dispatch_update;
use rmc_core::mc::{WeightedUpdate, WeightedUpdateSet};
use rmc_core::random::{exponential_sample_bounded, uniform_index};
use rmc_core::Result;
use slotmap::Key;

use rmc_frohlich::diagram::{draw_new_q_from_uniforms, VKey};
use rmc_frohlich::physics;
use rmc_frohlich::updates::{
    ChangeInternalTau, ChangeQDirection, ChangeTau, ChangeTopology, RescaleDiagram,
};
use rmc_frohlich::Diagram;

use crate::vertex::Reweight;

fn vec3(v: &Vector3<f64>) -> [f64; 3] {
    [v.x, v.y, v.z]
}

fn draw_new_q<R: Rng + ?Sized>(rng: &mut R) -> Vector3<f64> {
    draw_new_q_from_uniforms(rng.random(), rng.random(), rng.random())
}

fn random_vertex<R: Rng + ?Sized>(d: &Diagram, rng: &mut R) -> VKey {
    d.storage[uniform_index(rng, d.storage.len())]
}

fn random_arc<R: Rng + ?Sized>(d: &Diagram, rng: &mut R) -> (VKey, VKey) {
    let vertex1 = random_vertex(d, rng);
    let vertex2 = d.v(vertex1).link;
    if d.v(vertex2).tau < d.v(vertex1).tau {
        (vertex2, vertex1)
    } else {
        (vertex1, vertex2)
    }
}

/// Insert a phonon arc (Fröhlich [`rmc_frohlich::updates::phonon::AddPhonon`]),
/// reweighted: adding an arc with momentum `q` multiplies acceptance by `R(q)`.
#[derive(Clone, Debug, Default)]
pub struct AddPhonon {
    vertex1: VKey,
    vertex2: VKey,
    tau1: f64,
    tau2: f64,
    q: Vector3<f64>,
    reweight: Reweight,
}

impl AddPhonon {
    pub fn with_reweight(reweight: Reweight) -> Self {
        Self {
            reweight,
            ..Self::default()
        }
    }

    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            self.attempt_from_zero_fake(d, rng)
        } else {
            self.higher_order(d, rng)
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        if d.order == 0 {
            d.set_to_fake_order_one(self.q);
        } else {
            d.insert_arc_between(self.vertex1, self.vertex2, self.tau1, self.tau2, self.q);
        }
        debug_assert!(rmc_frohlich::sanity::check_sanity(d).is_ok());
    }

    fn attempt_from_zero_fake<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        self.q = draw_new_q(rng);
        physics::add_phonon_zero_ratio(d.alpha, d.tau(), vec3(&d.momentum_out()), vec3(&self.q))
            * self.reweight.factor(self.q)
    }

    fn higher_order<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order >= d.max_order {
            return 0.0;
        }

        let mut vertex1 = random_vertex(d, rng);
        while d.next(vertex1).is_null() {
            vertex1 = random_vertex(d, rng);
        }

        let next1 = d.next(vertex1);
        let delta_t = d.v(next1).tau - d.v(vertex1).tau;
        self.tau1 = d.v(vertex1).tau + rng.random::<f64>() * delta_t;
        if self.tau1 - d.v(vertex1).tau < Diagram::DELTA_TAU_LIMIT
            || d.v(next1).tau - self.tau1 < Diagram::DELTA_TAU_LIMIT
        {
            return 0.0;
        }

        self.q = draw_new_q(rng);
        let lambda = physics::phonon_lambda(vec3(&self.q));
        self.tau2 = exponential_sample_bounded(rng.random(), lambda, self.tau1, d.max_tau);
        if self.tau2 - self.tau1 < Diagram::DELTA_TAU_LIMIT {
            return 0.0;
        }

        let (p_mean, vertex2) = d.get_p_mean_between(self.tau1, self.tau2, vertex1);
        let prev2 = if vertex2.is_null() { d.tail } else { d.prev(vertex2) };
        let prev_tau = d.v(prev2).tau;
        if self.tau2 - prev_tau < Diagram::DELTA_TAU_LIMIT
            || (!vertex2.is_null() && d.v(vertex2).tau - self.tau2 < Diagram::DELTA_TAU_LIMIT)
        {
            return 0.0;
        }
        self.vertex1 = vertex1;
        self.vertex2 = vertex2;

        let tail_extension_exponent = if vertex2.is_null() {
            d.dispersion(&d.momentum_out()) * (self.tau2 - d.tau())
        } else {
            0.0
        };
        physics::add_phonon_higher_ratio(
            d.alpha,
            d.order,
            delta_t,
            vec3(&self.q),
            vec3(&p_mean),
            self.tau2 - self.tau1,
            tail_extension_exponent,
            d.max_tau - self.tau1,
        ) * self.reweight.factor(self.q)
    }
}

/// Remove a phonon arc (inverse of [`AddPhonon`]), reweighted: removing an arc
/// with momentum `q` divides acceptance by `R(q)`.
#[derive(Clone, Debug, Default)]
pub struct RemovePhonon {
    vertex1: VKey,
    vertex2: VKey,
    q: Vector3<f64>,
    reweight: Reweight,
}

impl RemovePhonon {
    pub fn with_reweight(reweight: Reweight) -> Self {
        Self {
            reweight,
            ..Self::default()
        }
    }

    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order == d.min_order {
            0.0
        } else if d.order == 1 {
            self.attempt_to_zero_fake(d)
        } else {
            self.higher_order(d, rng)
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        if d.order == 1 {
            d.clear_fake_order_one();
        } else {
            d.remove_arc(self.vertex1, self.vertex2);
        }
        debug_assert!(rmc_frohlich::sanity::check_sanity(d).is_ok());
    }

    fn attempt_to_zero_fake(&mut self, d: &Diagram) -> f64 {
        self.vertex1 = d.head;
        self.vertex2 = d.tail;
        self.q = d.v(d.head).q;
        physics::remove_phonon_zero_ratio(d.alpha, d.tau(), vec3(&d.momentum_out()), vec3(&self.q))
            / self.reweight.factor(self.q)
    }

    fn higher_order<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        let mut v = random_vertex(d, rng);
        while v == d.head || d.v(v).link == d.head {
            v = random_vertex(d, rng);
        }
        let mut left = v;
        let mut right = d.v(v).link;
        if d.v(left).tau > d.v(right).tau {
            std::mem::swap(&mut left, &mut right);
        }
        self.vertex1 = left;
        self.vertex2 = right;
        self.q = d.v(left).q;

        if d.order != 2 {
            let second_last = d.prev(d.tail);
            let mut k = d.next(left);
            while k != right {
                if d.v(k).phonons_above == 1 && k != second_last {
                    return 0.0;
                }
                k = d.next(k);
            }
        }

        let delta_t = -d.v(d.prev(left)).tau
            + if d.next(left) == right { d.v(d.next(right)).tau } else { d.v(d.next(left)).tau };
        let p_mean = d.get_p_mean_range(left, right, self.q);
        let tail_extension_exponent = if d.next(right).is_null() {
            d.dispersion(&d.momentum_out()) * (d.v(right).tau - d.v(d.prev(right)).tau)
        } else {
            0.0
        };
        physics::remove_phonon_higher_ratio(
            d.alpha,
            d.order,
            delta_t,
            vec3(&self.q),
            vec3(&p_mean),
            d.v(right).tau - d.v(left).tau,
            tail_extension_exponent,
            d.max_tau - d.v(left).tau,
        ) / self.reweight.factor(self.q)
    }
}

/// Resample an arc's `|q|` (Fröhlich [`rmc_frohlich::updates::ChangeQModulus`]),
/// reweighted: `q → q'` multiplies acceptance by `R(q')/R(q)`.
#[derive(Clone, Debug, Default)]
pub struct ChangeQModulus {
    vertex1: VKey,
    vertex2: VKey,
    q_prime: f64,
    reweight: Reweight,
}

impl ChangeQModulus {
    pub fn with_reweight(reweight: Reweight) -> Self {
        Self {
            reweight,
            ..Self::default()
        }
    }

    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        use rand_distr::{Distribution, Normal};
        if d.order == 0 {
            return 0.0;
        }
        let (left, right) = random_arc(d, rng);
        let q_vec = d.v(left).q;
        let q_norm = q_vec.norm();
        if q_norm == 0.0 {
            return 0.0;
        }
        let q0 = d
            .get_p_mean_range(left, right, q_vec)
            .dot(&(q_vec / q_norm));
        let sigma = physics::change_q_modulus_sigma(d.v(right).tau - d.v(left).tau);
        let Ok(normal) = Normal::new(q0, sigma) else {
            return 0.0;
        };
        self.q_prime = normal.sample(rng);
        self.vertex1 = left;
        self.vertex2 = right;
        if self.q_prime < 0.0 || !self.q_prime.is_finite() {
            return 0.0;
        }
        // Reweight the (Fröhlich-exact, acceptance-1) proposal to the target vertex.
        let new_q = q_vec / q_norm * self.q_prime;
        self.reweight.factor(new_q) / self.reweight.factor(q_vec)
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        let new_q = d.v(self.vertex1).q / d.v(self.vertex1).q.norm() * self.q_prime;
        d.update_arc_q(self.vertex1, self.vertex2, new_q);
        debug_assert!(rmc_frohlich::sanity::check_sanity(d).is_ok());
    }
}

dispatch_update! {
    #[derive(Clone, Debug)]
    pub enum AbInitioUpdate<Diagram> {
        ChangeTau(ChangeTau),
        ChangeInternalTau(ChangeInternalTau),
        AddPhonon(AddPhonon),
        RemovePhonon(RemovePhonon),
        RescaleDiagram(RescaleDiagram),
        ChangeQModulus(ChangeQModulus),
        ChangeQDirection(ChangeQDirection),
        ChangeTopology(ChangeTopology),
    }
}

/// The eight-move set with the material reweight installed on the three arc-`q`
/// moves. `Reweight::identity()` reproduces the Fröhlich engine bit-for-bit.
pub fn update_set(reweight: Reweight) -> Result<WeightedUpdateSet<AbInitioUpdate>> {
    WeightedUpdateSet::new(vec![
        WeightedUpdate::new(AbInitioUpdate::ChangeTau(ChangeTau::default()), 1.0),
        WeightedUpdate::new(
            AbInitioUpdate::ChangeInternalTau(ChangeInternalTau::default()),
            1.0,
        ),
        WeightedUpdate::new(
            AbInitioUpdate::AddPhonon(AddPhonon::with_reweight(reweight.clone())),
            1.0,
        ),
        WeightedUpdate::new(
            AbInitioUpdate::RemovePhonon(RemovePhonon::with_reweight(reweight.clone())),
            1.0,
        ),
        WeightedUpdate::new(
            AbInitioUpdate::RescaleDiagram(RescaleDiagram::default()),
            1.0,
        ),
        WeightedUpdate::new(
            AbInitioUpdate::ChangeQModulus(ChangeQModulus::with_reweight(reweight.clone())),
            1.0,
        ),
        WeightedUpdate::new(
            AbInitioUpdate::ChangeQDirection(ChangeQDirection::default()),
            1.0,
        ),
        WeightedUpdate::new(
            AbInitioUpdate::ChangeTopology(ChangeTopology::default()),
            1.0,
        ),
    ])
}
