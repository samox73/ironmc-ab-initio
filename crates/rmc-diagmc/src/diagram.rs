use nalgebra::Vector3;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::weight::LocalRatio;

const NONE: usize = usize::MAX;

/// One endpoint of a phonon arc, or an external endpoint (`head`/`tail`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Vertex {
    pub tau: f64,
    pub p_out: Vector3<f64>,
    pub q: Vector3<f64>,
    pub branch: usize,
    pub link: usize,
    pub prev: usize,
    pub next: usize,
    arc_id: usize,
}

impl Vertex {
    fn external(tau: f64, p: Vector3<f64>) -> Self {
        Self {
            tau,
            p_out: p,
            q: Vector3::zeros(),
            branch: 0,
            link: NONE,
            prev: NONE,
            next: NONE,
            arc_id: NONE,
        }
    }

    fn arc(tau: f64, q: Vector3<f64>, branch: usize, arc_id: usize) -> Self {
        let mut vertex = Self::external(tau, Vector3::zeros());
        vertex.q = q;
        vertex.branch = branch;
        vertex.arc_id = arc_id;
        vertex
    }
}

/// A time-ordered electron line with pairwise-linked phonon vertices.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Diagram {
    pub vertices: Vec<Vertex>,
    pub head: usize,
    pub tail: usize,
    pub order: usize,
    pub mu: f64,
    pub p_ext: Vector3<f64>,
    pub max_tau: f64,
    pub min_order: usize,
    pub max_order: usize,
    next_arc_id: usize,
    #[serde(default = "invalid_log_modulus")]
    cached_log_modulus: f64,
    #[serde(default = "unit_phase")]
    cached_phase: Complex64,
}

fn invalid_log_modulus() -> f64 {
    f64::NAN
}

fn unit_phase() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

impl Diagram {
    pub fn new(
        mu: f64,
        p_ext: Vector3<f64>,
        max_tau: f64,
        min_order: usize,
        max_order: usize,
    ) -> Self {
        assert!(max_tau > 0.0 && max_tau.is_finite());
        assert!(min_order <= max_order);
        let mut diagram = Self {
            vertices: vec![
                Vertex::external(0.0, p_ext),
                Vertex::external(max_tau / 2.0, p_ext),
            ],
            head: 0,
            tail: 1,
            order: 0,
            mu,
            p_ext,
            max_tau,
            min_order,
            max_order,
            next_arc_id: 0,
            cached_log_modulus: f64::NAN,
            cached_phase: unit_phase(),
        };
        diagram.patch_neighbors();
        diagram
    }

    pub fn tau(&self) -> f64 {
        self.vertices[self.tail].tau
    }

    pub fn v(&self, key: usize) -> &Vertex {
        &self.vertices[key]
    }

    pub fn ordered_keys(&self) -> std::ops::Range<usize> {
        0..self.vertices.len()
    }

    pub fn arc_keys(&self) -> Vec<(usize, usize)> {
        self.vertices
            .iter()
            .enumerate()
            .filter_map(|(i, vertex)| {
                (vertex.link > i && vertex.link != NONE).then_some((i, vertex.link))
            })
            .collect()
    }

    pub fn arc_count(&self) -> usize {
        self.order
    }

    pub fn nth_arc(&self, mut nth: usize) -> Option<(usize, usize)> {
        for (left, vertex) in self.vertices.iter().enumerate() {
            if vertex.link > left && vertex.link != NONE {
                if nth == 0 {
                    return Some((left, vertex.link));
                }
                nth -= 1;
            }
        }
        None
    }

    pub fn cached_phase(&self) -> Complex64 {
        self.cached_phase
    }

    pub fn cached_log_modulus(&self) -> f64 {
        self.cached_log_modulus
    }

    pub fn set_cached_weight(&mut self, log_modulus: f64, phase: Complex64) {
        self.cached_log_modulus = log_modulus;
        self.cached_phase = phase;
    }

    pub fn apply_ratio(&mut self, ratio: &LocalRatio) {
        self.cached_log_modulus += ratio.log_modulus;
        self.cached_phase *= ratio.phase;
        let norm = self.cached_phase.norm();
        if norm > 0.0 {
            self.cached_phase /= norm;
        }
    }

    pub fn incoming_momentum(&self, key: usize) -> Vector3<f64> {
        if key == self.head {
            self.p_ext
        } else {
            self.vertices[self.vertices[key].prev].p_out
        }
    }

    pub fn insert_arc(
        &mut self,
        tau1: f64,
        tau2: f64,
        q: Vector3<f64>,
        branch: usize,
    ) -> (usize, usize) {
        self.insert_arc_between(self.head, self.tail, tau1, tau2, q, branch)
    }

    pub fn insert_arc_between(
        &mut self,
        _left: usize,
        _before_right: usize,
        tau1: f64,
        tau2: f64,
        q: Vector3<f64>,
        branch: usize,
    ) -> (usize, usize) {
        assert!(0.0 < tau1 && tau1 < tau2 && tau2 < self.tau());
        let arc_id = self.next_arc_id;
        self.next_arc_id += 1;
        let left = self.vertices.partition_point(|vertex| vertex.tau < tau1);
        self.vertices
            .insert(left, Vertex::arc(tau1, q, branch, arc_id));
        for (key, vertex) in self.vertices.iter_mut().enumerate() {
            if key != left && vertex.link != NONE && vertex.link >= left {
                vertex.link += 1;
            }
        }

        let right = self.vertices.partition_point(|vertex| vertex.tau < tau2);
        self.vertices
            .insert(right, Vertex::arc(tau2, q, branch, arc_id));
        for (key, vertex) in self.vertices.iter_mut().enumerate() {
            if key != right && vertex.link != NONE && vertex.link >= right {
                vertex.link += 1;
            }
        }
        self.vertices[left].link = right;
        self.vertices[right].link = left;
        self.patch_neighbors();

        self.vertices[left].p_out = self.vertices[left - 1].p_out - q;
        for key in left + 1..right {
            self.vertices[key].p_out -= q;
        }
        self.vertices[right].p_out = self.vertices[right - 1].p_out + q;
        self.order += 1;
        debug_assert!(self.check_consistency());
        (left, right)
    }

    pub fn remove_arc(&mut self, a: usize, b: usize) {
        assert_eq!(self.vertices[a].link, b);
        assert_eq!(self.vertices[b].link, a);
        assert!(a < b);
        let q = self.vertices[a].q;
        self.remove_vertex(b);
        self.remove_vertex(a);
        for key in a..b - 1 {
            self.vertices[key].p_out += q;
        }
        self.patch_neighbors();
        self.order -= 1;
        debug_assert!(self.check_consistency());
    }

    pub fn update_arc_q(&mut self, a: usize, b: usize, q: Vector3<f64>) {
        assert_eq!(self.vertices[a].link, b);
        assert!(a < b);
        let delta = self.vertices[a].q - q;
        self.vertices[a].q = q;
        self.vertices[b].q = q;
        for key in a..b {
            self.vertices[key].p_out += delta;
        }
        debug_assert!(self.check_consistency());
    }

    pub fn set_vertex_tau(&mut self, key: usize, tau: f64) {
        assert!(key != self.head);
        assert!(self.vertices[key - 1].tau < tau);
        if key != self.tail {
            assert!(tau < self.vertices[key + 1].tau);
        } else {
            assert!(tau <= self.max_tau);
        }
        self.vertices[key].tau = tau;
        debug_assert!(self.check_consistency());
    }

    pub fn get_p_mean_range(
        &self,
        begin: usize,
        end: usize,
        addition: Vector3<f64>,
    ) -> Vector3<f64> {
        assert!(begin < end);
        let span = self.vertices[end].tau - self.vertices[begin].tau;
        let mut mean = Vector3::zeros();
        for key in begin..end {
            let dt = self.vertices[key + 1].tau - self.vertices[key].tau;
            mean += (self.vertices[key].p_out + addition) * (dt / span);
        }
        mean
    }

    pub fn get_p_mean_between(&self, tau1: f64, tau2: f64, begin: usize) -> (Vector3<f64>, usize) {
        assert!(tau1 < tau2);
        let mut sum = Vector3::zeros();
        let mut key = begin;
        while key < self.tail && self.vertices[key + 1].tau < tau2 {
            let left = tau1.max(self.vertices[key].tau);
            let right = tau2.min(self.vertices[key + 1].tau);
            sum += self.vertices[key].p_out * (right - left).max(0.0);
            key += 1;
        }
        let left = tau1.max(self.vertices[key].tau);
        sum += self.vertices[key].p_out * (tau2 - left).max(0.0);
        sum /= tau2 - tau1;
        (sum, (key + 1).min(self.tail))
    }

    fn patch_neighbors(&mut self) {
        self.head = 0;
        self.tail = self.vertices.len() - 1;
        for key in 0..self.vertices.len() {
            self.vertices[key].prev = key.checked_sub(1).unwrap_or(NONE);
            self.vertices[key].next = if key + 1 < self.vertices.len() { key + 1 } else { NONE };
        }
    }

    fn remove_vertex(&mut self, key: usize) {
        self.vertices.remove(key);
        for vertex in &mut self.vertices {
            if vertex.link == key {
                vertex.link = NONE;
            } else if vertex.link != NONE && vertex.link > key {
                vertex.link -= 1;
            }
        }
    }

    /// Full structural audit: index layout, τ ordering, link symmetry, and the
    /// momenta re-derived from every arc's q. Called via `debug_assert!` after
    /// each mutation, and periodically from the release-mode drift check in
    /// `run.rs` — the only guard against silent momentum-bookkeeping
    /// corruption in production runs.
    pub fn check_consistency(&self) -> bool {
        if self.head != 0
            || self.tail + 1 != self.vertices.len()
            || self.order * 2 + 2 != self.vertices.len()
            || self
                .vertices
                .windows(2)
                .any(|pair| pair[0].tau >= pair[1].tau)
        {
            return false;
        }
        let mut momentum = self.p_ext;
        for key in 0..self.vertices.len() {
            let vertex = &self.vertices[key];
            if vertex.prev != key.checked_sub(1).unwrap_or(NONE)
                || vertex.next != if key + 1 < self.vertices.len() { key + 1 } else { NONE }
            {
                return false;
            }
            if key == self.head || key == self.tail {
                if vertex.link != NONE {
                    return false;
                }
            } else {
                if vertex.link >= self.vertices.len()
                    || self.vertices[vertex.link].link != key
                    || self.vertices[vertex.link].arc_id != vertex.arc_id
                {
                    return false;
                }
                momentum += vertex.q * if vertex.link > key { -1.0 } else { 1.0 };
            }
            if (vertex.p_out - momentum).norm() >= 1.0e-10 {
                return false;
            }
        }
        (momentum - self.p_ext).norm() < 1.0e-10
    }

    /// Electron segments in τ order: `(outgoing momentum, Δτ)` for each of the
    /// `tail` propagator pieces. The single source of the segment-walk
    /// convention shared by the weight and the thermodynamic estimator.
    pub fn segments(&self) -> impl Iterator<Item = (Vector3<f64>, f64)> + '_ {
        (0..self.tail).map(|key| {
            (
                self.vertices[key].p_out,
                self.vertices[key + 1].tau - self.vertices[key].tau,
            )
        })
    }

    /// Phonon arcs as `(left, right)` vertex keys, without allocating.
    pub fn arcs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.vertices
            .iter()
            .enumerate()
            .filter_map(|(key, vertex)| {
                (vertex.link != NONE && vertex.link > key).then_some((key, vertex.link))
            })
    }
}
