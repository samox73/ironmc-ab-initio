//! Polaron Monte Carlo updates.

pub mod phonon;

use std::f64::consts::PI;

use nalgebra::Vector3;
use rand::Rng;
use rmc_core::dispatch_update;
use rmc_core::mc::{WeightedUpdate, WeightedUpdateSet};
use rmc_core::random::{normal_from_uniforms, safe_exponential_sample, uniform_index_from_u01};
use rmc_core::Result;
use slotmap::Key;

use crate::diagram::VKey;
use crate::diagram::{phi_from_cartesian, theta_from_cartesian, vec3, Diagram};
use crate::physics;
use phonon::{AddPhonon, RemovePhonon};

dispatch_update! {
    #[derive(Clone, Debug)]
    pub enum PolaronUpdate<Diagram> {
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

/// The full update set: all eight diagram moves, equally weighted.
pub fn default_update_set() -> Result<WeightedUpdateSet<PolaronUpdate>> {
    WeightedUpdateSet::new(vec![
        WeightedUpdate::new(PolaronUpdate::ChangeTau(ChangeTau::default()), 1.0),
        WeightedUpdate::new(
            PolaronUpdate::ChangeInternalTau(ChangeInternalTau::default()),
            1.0,
        ),
        WeightedUpdate::new(PolaronUpdate::AddPhonon(AddPhonon::default()), 1.0),
        WeightedUpdate::new(PolaronUpdate::RemovePhonon(RemovePhonon::default()), 1.0),
        WeightedUpdate::new(
            PolaronUpdate::RescaleDiagram(RescaleDiagram::default()),
            1.0,
        ),
        WeightedUpdate::new(
            PolaronUpdate::ChangeQModulus(ChangeQModulus::default()),
            1.0,
        ),
        WeightedUpdate::new(
            PolaronUpdate::ChangeQDirection(ChangeQDirection::default()),
            1.0,
        ),
        WeightedUpdate::new(
            PolaronUpdate::ChangeTopology(ChangeTopology::default()),
            1.0,
        ),
    ])
}

/// Resamples the diagram's total imaginary time (the tail vertex's tau).
#[derive(Clone, Debug, Default)]
pub struct ChangeTau {
    tau_prime: f64,
}

impl ChangeTau {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        let second_last = d.prev(d.tail);
        let lambda =
            d.dispersion(&d.v(second_last).p_out) + if d.order != 0 { physics::OMEGA } else { 0.0 };
        self.tau_prime =
            safe_exponential_sample(rng.gen(), lambda, d.v(second_last).tau, d.max_tau);
        // List order and arc orientation rely on strictly increasing taus; a sample landing
        // (up to rounding) on the neighbor's tau would create a degenerate pair.
        if self.tau_prime - d.v(second_last).tau < physics::DELTA_TAU_LIMIT {
            return 0.0;
        }
        1.0
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        d.set_vertex_tau(d.tail, self.tau_prime);
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

/// Resamples the tau of a randomly chosen internal (non-head/tail) vertex.
#[derive(Clone, Debug, Default)]
pub struct ChangeInternalTau {
    vertex: VKey,
    tau_prime: f64,
}

impl ChangeInternalTau {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order <= 1 {
            return 0.0;
        }
        let mut vertex = random_vertex(d, rng);
        while vertex == d.head || vertex == d.tail {
            vertex = random_vertex(d, rng);
        }
        let prev = d.prev(vertex);
        let next = d.next(vertex);
        let tau_previous = d.v(prev).tau;
        let tau_next = if next.is_null() {
            d.max_tau
        } else {
            d.v(next).tau
        };
        let lambda = physics::change_internal_tau_lambda(
            vec3(&d.v(prev).p_out),
            vec3(&d.v(vertex).p_out),
            d.is_incoming(vertex),
        );
        self.vertex = vertex;
        self.tau_prime = safe_exponential_sample(rng.gen(), lambda, tau_previous, tau_next);
        // Reject tau-degenerate proposals; strict ordering is load-bearing (see ChangeTau).
        if self.tau_prime.is_finite()
            && self.tau_prime - tau_previous >= physics::DELTA_TAU_LIMIT
            && tau_next - self.tau_prime >= physics::DELTA_TAU_LIMIT
        {
            1.0
        } else {
            0.0
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        d.set_vertex_tau(self.vertex, self.tau_prime);
        debug_assert!(d.v(d.prev(self.vertex)).tau < d.v(self.vertex).tau);
        let next = d.next(self.vertex);
        debug_assert!(next.is_null() || d.v(self.vertex).tau < d.v(next).tau);
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

/// Rescales all vertex taus by a common factor, proposed from the diagram's energy scale.
#[derive(Clone, Debug, Default)]
pub struct RescaleDiagram {
    tau_prime: f64,
}

impl RescaleDiagram {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order <= 1 {
            return 0.0;
        }

        let mut energy = -d.mu;
        let mut min_delta = f64::INFINITY;
        let mut k = d.head;
        while k != d.tail {
            let next = d.next(k);
            let delta = d.v(next).tau - d.v(k).tau;
            min_delta = min_delta.min(delta);
            let delta_s_i = delta / d.tau();
            let phonon_count = if d.is_incoming(k) {
                d.v(k).phonons_above
            } else {
                d.v(k).phonons_above + 1
            };
            energy += physics::rescale_energy_term(delta_s_i, vec3(&d.v(k).p_out), phonon_count);
            k = next;
        }

        let n = (d.order - 1) as f64;
        let sigma = (2.0 * n).sqrt() / energy;
        if !sigma.is_finite() || sigma < 0.0 {
            return 0.0;
        }
        self.tau_prime = normal_from_uniforms(rng.gen(), rng.gen(), 2.0 * n / energy, sigma);
        if self.tau_prime < 0.0 || self.tau_prime > d.max_tau || !self.tau_prime.is_finite() {
            return 0.0;
        }
        // Rescaling multiplies every tau by tau_prime / tau; with a 10x margin over
        // DELTA_TAU_LIMIT no adjacent pair can collapse to equal floats through rounding.
        if min_delta * (self.tau_prime / d.tau()) < 10.0 * physics::DELTA_TAU_LIMIT {
            return 0.0;
        }

        let acceptance = physics::rescale_diagram_ratio(d.order, d.tau(), self.tau_prime, energy);
        if acceptance.is_finite() {
            acceptance
        } else {
            0.0
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        d.scale_taus(self.tau_prime / d.tau());
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

/// Resamples the momentum-transfer magnitude `|q|` of a randomly chosen phonon arc.
#[derive(Clone, Debug, Default)]
pub struct ChangeQModulus {
    vertex1: VKey,
    vertex2: VKey,
    q_prime: f64,
}

impl ChangeQModulus {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            return 0.0;
        }
        let (left, right) = random_arc(d, rng);
        let q_norm = d.v(left).q.norm();
        if q_norm == 0.0 {
            return 0.0;
        }
        let q0 = d
            .get_p_mean_range(left, right, d.v(left).q)
            .dot(&(d.v(left).q / q_norm));
        let sigma = physics::change_q_modulus_sigma(d.v(right).tau - d.v(left).tau);
        if !sigma.is_finite() || sigma < 0.0 {
            return 0.0;
        }
        self.q_prime = normal_from_uniforms(rng.gen(), rng.gen(), q0, sigma);
        self.vertex1 = left;
        self.vertex2 = right;
        if self.q_prime < 0.0 || !self.q_prime.is_finite() {
            0.0
        } else {
            1.0
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        let new_q = d.v(self.vertex1).q / d.v(self.vertex1).q.norm() * self.q_prime;
        d.update_arc_q(self.vertex1, self.vertex2, new_q);
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

/// Resamples the direction of a phonon arc's momentum transfer `q`, keeping `|q|` fixed.
#[derive(Clone, Debug, Default)]
pub struct ChangeQDirection {
    vertex1: VKey,
    vertex2: VKey,
    q_prime: Vector3<f64>,
}

impl ChangeQDirection {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            return 0.0;
        }
        let (left, right) = random_arc(d, rng);
        let p_mean = d.get_p_mean_range(left, right, d.v(left).q);
        let q_norm = d.v(left).q.norm();
        let a =
            physics::change_q_direction_a(d.v(right).tau - d.v(left).tau, p_mean.norm(), q_norm);
        if a.abs() < f64::EPSILON || !a.is_finite() {
            return 0.0;
        }

        let phi = 2.0 * PI * rng.gen::<f64>();
        let log_val = (-2.0 * rng.gen::<f64>() * a.sinh() + a.exp()).ln();
        let theta = (log_val / a).acos();
        let theta_base = theta_from_cartesian(&p_mean);
        let phi_base = phi_from_cartesian(&p_mean);
        let rotated = physics::rotate_z_then_y(
            phi_base,
            theta_base,
            physics::spherical_to_cartesian(q_norm, theta, phi),
        );
        self.q_prime = Vector3::new(rotated[0], rotated[1], rotated[2]);
        self.vertex1 = left;
        self.vertex2 = right;
        if self.q_prime.iter().any(|x| x.is_nan()) {
            0.0
        } else {
            1.0
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        d.update_arc_q(self.vertex1, self.vertex2, self.q_prime);
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

/// Swaps the phonon-arc connectivity between two adjacent vertices, changing which arcs cross
/// which time interval without adding or removing vertices.
#[derive(Clone, Debug, Default)]
pub struct ChangeTopology {
    vertex1: VKey,
    vertex2: VKey,
    p_prime: Vector3<f64>,
}

impl ChangeTopology {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            return 0.0;
        }
        let mut vertex1 = random_vertex(d, rng);
        if d.next(vertex1).is_null() {
            vertex1 = d.prev(vertex1);
        }
        let vertex2 = d.next(vertex1);
        let c1 = if d.is_outgoing(vertex1) { 1.0 } else { -1.0 };
        let c2 = if d.is_outgoing(vertex2) { 1.0 } else { -1.0 };
        if d.v(vertex1).link == vertex2 {
            return 0.0;
        }
        self.p_prime = d.v(vertex1).p_out + d.v(vertex1).q * c1 - d.v(vertex2).q * c2;

        if d.is_outgoing(vertex1)
            && d.is_incoming(vertex2)
            && d.v(vertex1).link != vertex2
            && (d.v(vertex1).phonons_above == 1 || d.v(vertex2).phonons_above == 1)
        {
            return 0.0;
        }

        self.vertex1 = vertex1;
        self.vertex2 = vertex2;
        let acceptance = physics::change_topology_ratio(
            d.v(vertex2).tau - d.v(vertex1).tau,
            vec3(&self.p_prime),
            vec3(&d.v(vertex1).p_out),
            c1,
            c2,
        );
        if acceptance.is_finite() {
            acceptance
        } else {
            0.0
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        if d.is_incoming(self.vertex1) && d.is_outgoing(self.vertex2) {
            d.vm(self.vertex1).phonons_above += 1;
            d.vm(self.vertex2).phonons_above += 1;
        }
        if d.is_outgoing(self.vertex1)
            && d.is_incoming(self.vertex2)
            && d.v(self.vertex1).link != self.vertex2
        {
            d.vm(self.vertex1).phonons_above -= 1;
            d.vm(self.vertex2).phonons_above -= 1;
        }
        d.vm(self.vertex1).p_out = self.p_prime;
        if d.v(self.vertex1).link != self.vertex2 {
            d.swap_arc_connectivity(self.vertex1, self.vertex2);
        }
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

fn random_vertex<R: Rng + ?Sized>(d: &Diagram, rng: &mut R) -> VKey {
    d.storage[uniform_index_from_u01(rng.gen(), d.storage.len())]
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
