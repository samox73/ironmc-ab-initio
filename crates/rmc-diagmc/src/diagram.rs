use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

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
        };
        diagram.rebuild();
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
        self.vertices.push(Vertex::arc(tau1, q, branch, arc_id));
        self.vertices.push(Vertex::arc(tau2, q, branch, arc_id));
        self.order += 1;
        self.rebuild();
        self.arc_keys()
            .into_iter()
            .find(|&(left, _)| self.vertices[left].arc_id == arc_id)
            .expect("inserted arc must exist")
    }

    pub fn remove_arc(&mut self, a: usize, b: usize) {
        assert_eq!(self.vertices[a].link, b);
        assert_eq!(self.vertices[b].link, a);
        let arc_id = self.vertices[a].arc_id;
        self.vertices.retain(|vertex| vertex.arc_id != arc_id);
        self.order -= 1;
        self.rebuild();
    }

    pub fn update_arc_q(&mut self, a: usize, b: usize, q: Vector3<f64>) {
        assert_eq!(self.vertices[a].link, b);
        let arc_id = self.vertices[a].arc_id;
        for vertex in &mut self.vertices {
            if vertex.arc_id == arc_id {
                vertex.q = q;
            }
        }
        self.rebuild();
    }

    pub fn set_vertex_tau(&mut self, key: usize, tau: f64) {
        self.vertices[key].tau = tau;
        self.rebuild();
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

    fn rebuild(&mut self) {
        self.vertices.sort_by(|a, b| a.tau.total_cmp(&b.tau));
        self.head = 0;
        self.tail = self.vertices.len() - 1;

        for key in 0..self.vertices.len() {
            self.vertices[key].prev = key.checked_sub(1).unwrap_or(NONE);
            self.vertices[key].next = if key + 1 < self.vertices.len() { key + 1 } else { NONE };
            self.vertices[key].link = NONE;
        }
        for left in 1..self.tail {
            if self.vertices[left].link != NONE {
                continue;
            }
            let right = (left + 1..self.tail)
                .find(|&right| self.vertices[right].arc_id == self.vertices[left].arc_id)
                .expect("every arc has two endpoints");
            self.vertices[left].link = right;
            self.vertices[right].link = left;
        }

        let mut p = self.p_ext;
        self.vertices[self.head].p_out = p;
        for key in 1..self.tail {
            let sign = if self.vertices[key].link > key { -1.0 } else { 1.0 };
            p += self.vertices[key].q * sign;
            self.vertices[key].p_out = p;
        }
        self.vertices[self.tail].p_out = p;
        debug_assert!((p - self.p_ext).norm() < 1.0e-10);
    }
}
