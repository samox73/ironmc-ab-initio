use std::f64::consts::PI;

use rand::Rng;
use rmc_core::dispatch_update;
use rmc_core::mc::{WeightedUpdate, WeightedUpdateSet};
use rmc_core::random::{
    exponential_sample_bounded, normal_from_uniforms, safe_exponential_sample,
    uniform_index_from_u01,
};
use rmc_core::Result;

use crate::flat::{add_vec, scale_vec, sub_vec, unit, FlatDiagram, NULL};
use crate::physics::{self, Vec3};

dispatch_update! {
    #[derive(Clone, Debug)]
    pub enum FlatPolaronUpdate<FlatDiagram> {
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

pub fn default_update_set() -> Result<WeightedUpdateSet<FlatPolaronUpdate>> {
    WeightedUpdateSet::new(vec![
        WeightedUpdate::new(FlatPolaronUpdate::ChangeTau(ChangeTau::default()), 1.0),
        WeightedUpdate::new(
            FlatPolaronUpdate::ChangeInternalTau(ChangeInternalTau::default()),
            1.0,
        ),
        WeightedUpdate::new(FlatPolaronUpdate::AddPhonon(AddPhonon::default()), 1.0),
        WeightedUpdate::new(
            FlatPolaronUpdate::RemovePhonon(RemovePhonon::default()),
            1.0,
        ),
        WeightedUpdate::new(
            FlatPolaronUpdate::RescaleDiagram(RescaleDiagram::default()),
            1.0,
        ),
        WeightedUpdate::new(
            FlatPolaronUpdate::ChangeQModulus(ChangeQModulus::default()),
            1.0,
        ),
        WeightedUpdate::new(
            FlatPolaronUpdate::ChangeQDirection(ChangeQDirection::default()),
            1.0,
        ),
        WeightedUpdate::new(
            FlatPolaronUpdate::ChangeTopology(ChangeTopology::default()),
            1.0,
        ),
    ])
}

#[derive(Clone, Debug, Default)]
pub struct ChangeTau {
    tau_prime: f64,
}

impl ChangeTau {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &FlatDiagram, rng: &mut R) -> f64 {
        let second_last = d.prev[d.tail as usize];
        let lambda = physics::dispersion(d.p_out[second_last as usize], d.mu)
            + if d.order != 0 { physics::OMEGA } else { 0.0 };
        self.tau_prime =
            safe_exponential_sample(rng.gen(), lambda, d.tau[second_last as usize], d.max_tau);
        // List order and arc orientation rely on strictly increasing taus; a sample landing
        // (up to rounding) on the neighbor's tau would create a degenerate pair.
        if self.tau_prime - d.tau[second_last as usize] < physics::DELTA_TAU_LIMIT {
            return 0.0;
        }
        1.0
    }

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        d.set_vertex_tau(d.tail, self.tau_prime);
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChangeInternalTau {
    vertex: u32,
    tau_prime: f64,
}

impl ChangeInternalTau {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &FlatDiagram, rng: &mut R) -> f64 {
        if d.order <= 1 {
            return 0.0;
        }
        let mut vertex = random_vertex(d, rng);
        while vertex == d.head || vertex == d.tail {
            vertex = random_vertex(d, rng);
        }
        let prev = d.prev[vertex as usize];
        let next = d.next[vertex as usize];
        let tau_previous = d.tau[prev as usize];
        let tau_next = if next == NULL {
            d.max_tau
        } else {
            d.tau[next as usize]
        };
        let lambda = physics::change_internal_tau_lambda(
            d.p_out[prev as usize],
            d.p_out[vertex as usize],
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

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        d.set_vertex_tau(self.vertex, self.tau_prime);
    }
}

#[derive(Clone, Debug, Default)]
pub struct AddPhonon {
    vertex1: u32,
    vertex2: u32,
    tau1: f64,
    tau2: f64,
    q: Vec3,
}

impl AddPhonon {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &FlatDiagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            self.attempt_from_zero_fake(d, rng)
        } else {
            self.higher_order(d, rng)
        }
    }

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        if d.order == 0 {
            d.set_to_fake_order_one(self.q);
        } else {
            d.insert_arc_between(self.vertex1, self.vertex2, self.tau1, self.tau2, self.q)
                .expect("accepted flat add proposal must have capacity");
        }
    }

    fn attempt_from_zero_fake<R: Rng + ?Sized>(&mut self, d: &FlatDiagram, rng: &mut R) -> f64 {
        self.q = draw_new_q(rng);
        physics::add_phonon_zero_ratio(d.alpha, d.tau(), d.momentum_out(), self.q)
    }

    fn higher_order<R: Rng + ?Sized>(&mut self, d: &FlatDiagram, rng: &mut R) -> f64 {
        if d.order >= d.max_order {
            return 0.0;
        }
        if !d.has_arc_capacity() {
            return -1.0;
        }

        let mut vertex1 = random_vertex(d, rng);
        while d.next[vertex1 as usize] == NULL {
            vertex1 = random_vertex(d, rng);
        }

        let next1 = d.next[vertex1 as usize];
        let delta_t = d.tau[next1 as usize] - d.tau[vertex1 as usize];
        self.tau1 = d.tau[vertex1 as usize] + rng.gen::<f64>() * delta_t;
        if self.tau1 - d.tau[vertex1 as usize] < physics::DELTA_TAU_LIMIT
            || d.tau[next1 as usize] - self.tau1 < physics::DELTA_TAU_LIMIT
        {
            return 0.0;
        }

        self.q = draw_new_q(rng);
        let lambda = physics::phonon_lambda(self.q);
        self.tau2 = exponential_sample_bounded(rng.gen(), lambda, self.tau1, d.max_tau);
        if self.tau2 - self.tau1 < physics::DELTA_TAU_LIMIT {
            return 0.0;
        }

        let (p_mean, vertex2) = d.get_p_mean_between(self.tau1, self.tau2, vertex1);
        let prev2 = if vertex2 == NULL {
            d.tail
        } else {
            d.prev[vertex2 as usize]
        };
        let prev_tau = d.tau[prev2 as usize];
        if self.tau2 - prev_tau < physics::DELTA_TAU_LIMIT
            || (vertex2 != NULL && d.tau[vertex2 as usize] - self.tau2 < physics::DELTA_TAU_LIMIT)
        {
            return 0.0;
        }
        self.vertex1 = vertex1;
        self.vertex2 = vertex2;

        let tail_extension_exponent = if vertex2 == NULL {
            physics::dispersion(d.momentum_out(), d.mu) * (self.tau2 - d.tau())
        } else {
            0.0
        };
        physics::add_phonon_higher_ratio(
            d.alpha,
            d.order,
            delta_t,
            self.q,
            p_mean,
            self.tau2 - self.tau1,
            tail_extension_exponent,
            d.max_tau - self.tau1,
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct RemovePhonon {
    vertex1: u32,
    vertex2: u32,
    q: Vec3,
}

impl RemovePhonon {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &FlatDiagram, rng: &mut R) -> f64 {
        if d.order == d.min_order {
            0.0
        } else if d.order == 1 {
            self.attempt_to_zero_fake(d)
        } else {
            self.higher_order(d, rng)
        }
    }

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        if d.order == 1 {
            d.clear_fake_order_one();
        } else {
            d.remove_arc(self.vertex1, self.vertex2);
        }
    }

    fn attempt_to_zero_fake(&mut self, d: &FlatDiagram) -> f64 {
        self.vertex1 = d.head;
        self.vertex2 = d.tail;
        self.q = d.q[d.head as usize];
        physics::remove_phonon_zero_ratio(d.alpha, d.tau(), d.momentum_out(), self.q)
    }

    fn higher_order<R: Rng + ?Sized>(&mut self, d: &FlatDiagram, rng: &mut R) -> f64 {
        let mut v = random_vertex(d, rng);
        while v == d.head || d.link[v as usize] == d.head {
            v = random_vertex(d, rng);
        }
        let mut left = v;
        let mut right = d.link[v as usize];
        if d.tau[left as usize] > d.tau[right as usize] {
            std::mem::swap(&mut left, &mut right);
        }
        self.vertex1 = left;
        self.vertex2 = right;
        self.q = d.q[left as usize];

        if d.order != 2 {
            let second_last = d.prev[d.tail as usize];
            let mut slot = d.next[left as usize];
            while slot != right {
                if d.phonons_above[slot as usize] == 1 && slot != second_last {
                    return 0.0;
                }
                slot = d.next[slot as usize];
            }
        }

        let delta_t = -d.tau[d.prev[left as usize] as usize]
            + if d.next[left as usize] == right {
                d.tau[d.next[right as usize] as usize]
            } else {
                d.tau[d.next[left as usize] as usize]
            };
        let p_mean = d.get_p_mean_range(left, right, self.q);
        let tail_extension_exponent = if d.next[right as usize] == NULL {
            physics::dispersion(d.momentum_out(), d.mu)
                * (d.tau[right as usize] - d.tau[d.prev[right as usize] as usize])
        } else {
            0.0
        };
        physics::remove_phonon_higher_ratio(
            d.alpha,
            d.order,
            delta_t,
            self.q,
            p_mean,
            d.tau[right as usize] - d.tau[left as usize],
            tail_extension_exponent,
            d.max_tau - d.tau[left as usize],
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct RescaleDiagram {
    tau_prime: f64,
}

impl RescaleDiagram {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &FlatDiagram, rng: &mut R) -> f64 {
        if d.order <= 1 {
            return 0.0;
        }

        let mut energy = -d.mu;
        let mut min_delta = f64::INFINITY;
        let mut slot = d.head;
        while slot != d.tail {
            let next = d.next[slot as usize];
            let delta = d.tau[next as usize] - d.tau[slot as usize];
            min_delta = min_delta.min(delta);
            let delta_s_i = delta / d.tau();
            let phonon_count = if d.is_incoming(slot) {
                d.phonons_above[slot as usize] as usize
            } else {
                d.phonons_above[slot as usize] as usize + 1
            };
            energy += physics::rescale_energy_term(delta_s_i, d.p_out[slot as usize], phonon_count);
            slot = next;
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

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        d.scale_taus(self.tau_prime / d.tau());
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChangeQModulus {
    vertex1: u32,
    vertex2: u32,
    q_prime: f64,
}

impl ChangeQModulus {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &FlatDiagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            return 0.0;
        }
        let (left, right) = random_arc(d, rng);
        let q_norm = physics::norm(d.q[left as usize]);
        if q_norm == 0.0 {
            return 0.0;
        }
        let q0 = physics::dot(
            d.get_p_mean_range(left, right, d.q[left as usize]),
            scale_vec(d.q[left as usize], 1.0 / q_norm),
        );
        let sigma = physics::change_q_modulus_sigma(d.tau[right as usize] - d.tau[left as usize]);
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

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        let new_q = scale_vec(unit(d.q[self.vertex1 as usize]), self.q_prime);
        d.update_arc_q(self.vertex1, self.vertex2, new_q);
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChangeQDirection {
    vertex1: u32,
    vertex2: u32,
    q_prime: Vec3,
}

impl ChangeQDirection {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &FlatDiagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            return 0.0;
        }
        let (left, right) = random_arc(d, rng);
        let p_mean = d.get_p_mean_range(left, right, d.q[left as usize]);
        let q_norm = physics::norm(d.q[left as usize]);
        let a = physics::change_q_direction_a(
            d.tau[right as usize] - d.tau[left as usize],
            physics::norm(p_mean),
            q_norm,
        );
        if a.abs() < f64::EPSILON || !a.is_finite() {
            return 0.0;
        }

        let phi = 2.0 * PI * rng.gen::<f64>();
        let log_val = (-2.0 * rng.gen::<f64>() * a.sinh() + a.exp()).ln();
        let theta = (log_val / a).acos();
        let theta_base = physics::theta_from_cartesian(p_mean);
        let phi_base = physics::phi_from_cartesian(p_mean);
        self.q_prime = physics::rotate_z_then_y(
            phi_base,
            theta_base,
            physics::spherical_to_cartesian(q_norm, theta, phi),
        );
        self.vertex1 = left;
        self.vertex2 = right;
        if self.q_prime.iter().any(|x| x.is_nan()) {
            0.0
        } else {
            1.0
        }
    }

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        d.update_arc_q(self.vertex1, self.vertex2, self.q_prime);
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChangeTopology {
    vertex1: u32,
    vertex2: u32,
    p_prime: Vec3,
}

impl ChangeTopology {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &FlatDiagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            return 0.0;
        }
        let mut vertex1 = random_vertex(d, rng);
        if d.next[vertex1 as usize] == NULL {
            vertex1 = d.prev[vertex1 as usize];
        }
        let vertex2 = d.next[vertex1 as usize];
        let c1 = if d.is_outgoing(vertex1) { 1.0 } else { -1.0 };
        let c2 = if d.is_outgoing(vertex2) { 1.0 } else { -1.0 };
        if d.link[vertex1 as usize] == vertex2 {
            return 0.0;
        }
        self.p_prime = sub_vec(
            add_vec(
                d.p_out[vertex1 as usize],
                scale_vec(d.q[vertex1 as usize], c1),
            ),
            scale_vec(d.q[vertex2 as usize], c2),
        );

        if d.is_outgoing(vertex1)
            && d.is_incoming(vertex2)
            && d.link[vertex1 as usize] != vertex2
            && (d.phonons_above[vertex1 as usize] == 1 || d.phonons_above[vertex2 as usize] == 1)
        {
            return 0.0;
        }

        self.vertex1 = vertex1;
        self.vertex2 = vertex2;
        let acceptance = physics::change_topology_ratio(
            d.tau[vertex2 as usize] - d.tau[vertex1 as usize],
            self.p_prime,
            d.p_out[vertex1 as usize],
            c1,
            c2,
        );
        if acceptance.is_finite() {
            acceptance
        } else {
            0.0
        }
    }

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        if d.is_incoming(self.vertex1) && d.is_outgoing(self.vertex2) {
            d.phonons_above[self.vertex1 as usize] += 1;
            d.phonons_above[self.vertex2 as usize] += 1;
        }
        if d.is_outgoing(self.vertex1)
            && d.is_incoming(self.vertex2)
            && d.link[self.vertex1 as usize] != self.vertex2
        {
            d.phonons_above[self.vertex1 as usize] -= 1;
            d.phonons_above[self.vertex2 as usize] -= 1;
        }
        d.p_out[self.vertex1 as usize] = self.p_prime;
        if d.link[self.vertex1 as usize] != self.vertex2 {
            d.swap_arc_connectivity(self.vertex1, self.vertex2);
        }
    }
}

fn draw_new_q<R: Rng + ?Sized>(rng: &mut R) -> Vec3 {
    physics::draw_new_q_from_uniforms(rng.gen(), rng.gen(), rng.gen())
}

fn random_vertex<R: Rng + ?Sized>(d: &FlatDiagram, rng: &mut R) -> u32 {
    d.storage[uniform_index_from_u01(rng.gen(), d.storage.len())]
}

fn random_arc<R: Rng + ?Sized>(d: &FlatDiagram, rng: &mut R) -> (u32, u32) {
    let vertex1 = random_vertex(d, rng);
    let vertex2 = d.link[vertex1 as usize];
    if d.tau[vertex2 as usize] < d.tau[vertex1 as usize] {
        (vertex2, vertex1)
    } else {
        (vertex1, vertex2)
    }
}

