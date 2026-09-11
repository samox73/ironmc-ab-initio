//! Phase A2: fixed-capacity flat ab-initio diagram — the executable spec for
//! the GPU kernel.
//!
//! Unlike the Fröhlich flat engine (linked slots), the four `rmc-diagmc`
//! updates lean on τ-ordered *positions* everywhere (`segment_at`,
//! `partition_point` windows, `key ± 1` neighbors), so this layout is a dense
//! τ-sorted SoA array with positional keys — a direct transcription of
//! `rmc_diagmc::Diagram` onto u32 grid momenta. Inserts/removals shift at most
//! `2·order` fixed-size entries, cheaper than the list walks they replace at
//! LiF's mean orders. Momenta are exact flat grid indices (`wrap_add`/`neg`
//! integer arithmetic); fractional `Vector3` appears only in the adapters.

pub mod batched;
pub mod updates;
pub mod weight;

use nalgebra::Vector3;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::vertex_table::VertexTable;
use rmc_diagmc::weight::LocalRatio;

pub const NONE: u32 = u32::MAX;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlatDiagram {
    // SoA, τ-sorted, live prefix `len`; slot 0 = head, slot len−1 = tail.
    pub tau: Vec<f64>,
    /// Outgoing electron momentum per vertex, as a flat grid index.
    pub p_out: Vec<u32>,
    pub q: Vec<u32>,
    pub branch: Vec<u32>,
    pub link: Vec<u32>,
    pub len: usize,
    pub order: usize,
    pub grid_n: u32,
    pub mu: f64,
    pub p_ext: u32,
    pub max_tau: f64,
    pub min_order: usize,
    pub max_order: usize,
    /// Per-chain weight cache: sample |w|, carry phase separately.
    pub log_modulus: f64,
    pub phase: Complex64,
}

/// `(a + b) mod grid`, on flat indices. Mirrors `VertexTable::wrap_add`;
/// duplicated here because the flat diagram (like the kernel) carries only
/// `grid_n`, not the table — cross-checked in tests.
#[inline]
pub fn wrap_add(n: u32, a: u32, b: u32) -> u32 {
    let nn = n * n;
    let i = (a / nn + b / nn) % n;
    let j = (a / n % n + b / n % n) % n;
    let l = (a % n + b % n) % n;
    (i * n + j) * n + l
}

/// `−a mod grid` on flat indices (mirror of `VertexTable::neg`).
#[inline]
pub fn neg(n: u32, a: u32) -> u32 {
    let nn = n * n;
    let i = (n - a / nn % n) % n;
    let j = (n - a / n % n) % n;
    let l = (n - a % n) % n;
    (i * n + j) * n + l
}

#[inline]
pub fn wrap_sub(n: u32, a: u32, b: u32) -> u32 {
    wrap_add(n, a, neg(n, b))
}

impl FlatDiagram {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grid_n: u32,
        mu: f64,
        p_ext: u32,
        max_tau: f64,
        start_tau: f64,
        min_order: usize,
        max_order: usize,
        max_order_gpu: usize,
    ) -> Self {
        assert!(max_tau > 0.0 && max_tau.is_finite());
        assert!(0.0 < start_tau && start_tau <= max_tau);
        assert!(min_order <= max_order && max_order_gpu > 0);
        let capacity = 2 * max_order_gpu + 2;
        let mut diagram = Self {
            tau: vec![0.0; capacity],
            p_out: vec![0; capacity],
            q: vec![0; capacity],
            branch: vec![0; capacity],
            link: vec![NONE; capacity],
            len: 2,
            order: 0,
            grid_n,
            mu,
            p_ext,
            max_tau,
            min_order,
            max_order: max_order.min(max_order_gpu),
            log_modulus: f64::NAN,
            phase: Complex64::new(1.0, 0.0),
        };
        diagram.tau[1] = start_tau;
        diagram.p_out[0] = p_ext;
        diagram.p_out[1] = p_ext;
        diagram
    }

    #[inline]
    pub fn tail(&self) -> usize {
        self.len - 1
    }

    #[inline]
    pub fn tau_ext(&self) -> f64 {
        self.tau[self.tail()]
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.tau.len()
    }

    /// Incoming electron momentum at `key` (grid index).
    #[inline]
    pub fn incoming_momentum(&self, key: usize) -> u32 {
        if key == 0 {
            self.p_ext
        } else {
            self.p_out[key - 1]
        }
    }

    /// Last vertex with τ_v ≤ τ, clamped to tail−1 (mirrors `weight::segment_at`).
    pub fn segment_at(&self, tau: f64) -> usize {
        self.tau[..self.len]
            .partition_point(|&t| t <= tau)
            .saturating_sub(1)
            .min(self.tail() - 1)
    }

    /// Phonon arcs as `(left, right)` positional keys.
    pub fn arcs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.len).filter_map(|key| {
            let link = self.link[key];
            (link != NONE && link as usize > key).then_some((key, link as usize))
        })
    }

    /// The n-th arc ordered by left-endpoint position (mirrors `Diagram::nth_arc`).
    pub fn nth_arc(&self, mut nth: usize) -> Option<(usize, usize)> {
        for (left, right) in self.arcs() {
            if nth == 0 {
                return Some((left, right));
            }
            nth -= 1;
        }
        let _ = nth;
        None
    }

    fn shift_up(&mut self, at: usize) {
        assert!(self.len < self.capacity(), "flat diagram is at capacity");
        for key in (at..self.len).rev() {
            self.tau[key + 1] = self.tau[key];
            self.p_out[key + 1] = self.p_out[key];
            self.q[key + 1] = self.q[key];
            self.branch[key + 1] = self.branch[key];
            self.link[key + 1] = self.link[key];
        }
        for key in 0..=self.len {
            if key != at && self.link[key] != NONE && self.link[key] as usize >= at {
                self.link[key] += 1;
            }
        }
        self.len += 1;
    }

    fn shift_down(&mut self, at: usize) {
        for key in at..self.len - 1 {
            self.tau[key] = self.tau[key + 1];
            self.p_out[key] = self.p_out[key + 1];
            self.q[key] = self.q[key + 1];
            self.branch[key] = self.branch[key + 1];
            self.link[key] = self.link[key + 1];
        }
        self.len -= 1;
        for key in 0..self.len {
            if self.link[key] != NONE {
                if self.link[key] as usize == at {
                    self.link[key] = NONE;
                } else if self.link[key] as usize > at {
                    self.link[key] -= 1;
                }
            }
        }
    }

    /// Insert a phonon arc; mirrors `Diagram::insert_arc` including the
    /// momentum bookkeeping (`p → p − q` on the covered segments).
    pub fn insert_arc(&mut self, tau1: f64, tau2: f64, q: u32, branch: u32) -> (usize, usize) {
        assert!(0.0 < tau1 && tau1 < tau2 && tau2 < self.tau_ext());
        let n = self.grid_n;
        let left = self.tau[..self.len].partition_point(|&t| t < tau1);
        self.shift_up(left);
        self.tau[left] = tau1;
        self.q[left] = q;
        self.branch[left] = branch;
        self.link[left] = NONE;

        let right = self.tau[..self.len].partition_point(|&t| t < tau2);
        self.shift_up(right);
        self.tau[right] = tau2;
        self.q[right] = q;
        self.branch[right] = branch;
        self.link[left] = right as u32;
        self.link[right] = left as u32;

        self.p_out[left] = wrap_sub(n, self.p_out[left - 1], q);
        for key in left + 1..right {
            self.p_out[key] = wrap_sub(n, self.p_out[key], q);
        }
        self.p_out[right] = wrap_add(n, self.p_out[right - 1], q);
        self.order += 1;
        debug_assert!(self.check_consistency());
        (left, right)
    }

    /// Remove the arc `(a, b)`; mirrors `Diagram::remove_arc`.
    pub fn remove_arc(&mut self, a: usize, b: usize) {
        assert_eq!(self.link[a] as usize, b);
        assert_eq!(self.link[b] as usize, a);
        assert!(a < b);
        let n = self.grid_n;
        let q = self.q[a];
        self.shift_down(b);
        self.shift_down(a);
        for key in a..b - 1 {
            self.p_out[key] = wrap_add(n, self.p_out[key], q);
        }
        self.order -= 1;
        debug_assert!(self.check_consistency());
    }

    pub fn set_vertex_tau(&mut self, key: usize, tau: f64) {
        assert!(key != 0);
        assert!(self.tau[key - 1] < tau);
        if key != self.tail() {
            assert!(tau < self.tau[key + 1]);
        } else {
            assert!(tau <= self.max_tau);
        }
        self.tau[key] = tau;
        debug_assert!(self.check_consistency());
    }

    pub fn apply_ratio(&mut self, ratio: &LocalRatio) {
        self.log_modulus += ratio.log_modulus;
        self.phase *= ratio.phase;
        let norm = self.phase.norm();
        if norm > 0.0 {
            self.phase /= norm;
        }
    }

    pub fn set_cached_weight(&mut self, log_modulus: f64, phase: Complex64) {
        self.log_modulus = log_modulus;
        self.phase = phase;
    }

    /// Structural audit mirroring `Diagram::check_consistency`: τ ordering,
    /// link symmetry, and momenta re-derived exactly (integer arithmetic).
    pub fn check_consistency(&self) -> bool {
        if self.len < 2
            || self.order * 2 + 2 != self.len
            || self.tau[..self.len].windows(2).any(|w| w[0] >= w[1])
            || self.tau[0] != 0.0
        {
            return false;
        }
        let n = self.grid_n;
        let mut momentum = self.p_ext;
        for key in 0..self.len {
            let link = self.link[key];
            if key == 0 || key == self.tail() {
                if link != NONE {
                    return false;
                }
            } else {
                if link == NONE
                    || link as usize >= self.len
                    || self.link[link as usize] as usize != key
                    || self.q[link as usize] != self.q[key]
                    || self.branch[link as usize] != self.branch[key]
                {
                    return false;
                }
                momentum = if link as usize > key {
                    wrap_sub(n, momentum, self.q[key])
                } else {
                    wrap_add(n, momentum, self.q[key])
                };
            }
            if self.p_out[key] != momentum {
                return false;
            }
        }
        momentum == self.p_ext
    }

    // -- Adapters (index ↔ fractional) --------------------------------------

    /// Import a slotmap diagram; every momentum must sit on the table grid.
    pub fn from_diagram(
        diagram: &rmc_diagmc::Diagram,
        table: &VertexTable,
        max_order_gpu: usize,
    ) -> Self {
        let index = |v: Vector3<f64>| -> u32 {
            table.index_of(v).expect("momentum off the table grid") as u32
        };
        let mut flat = Self::new(
            table.grid_n as u32,
            diagram.mu,
            index(diagram.p_ext),
            diagram.max_tau,
            diagram.tau(),
            diagram.min_order,
            diagram.max_order,
            max_order_gpu,
        );
        assert!(diagram.vertices.len() <= flat.capacity());
        flat.len = diagram.vertices.len();
        flat.order = diagram.order;
        for (key, vertex) in diagram.vertices.iter().enumerate() {
            flat.tau[key] = vertex.tau;
            flat.p_out[key] = index(vertex.p_out);
            flat.q[key] = index(vertex.q);
            flat.branch[key] = vertex.branch as u32;
            flat.link[key] = if vertex.link == usize::MAX { NONE } else { vertex.link as u32 };
        }
        flat.log_modulus = diagram.cached_log_modulus();
        flat.phase = diagram.cached_phase();
        assert!(flat.check_consistency());
        flat
    }

    /// Export to a slotmap diagram (rebuilt through the public mutation API,
    /// so its own bookkeeping re-derives all momenta).
    pub fn to_diagram(&self, table: &VertexTable) -> rmc_diagmc::Diagram {
        let mut diagram = rmc_diagmc::Diagram::new(
            self.mu,
            table.point(self.p_ext as usize),
            self.max_tau,
            self.min_order,
            self.max_order,
        );
        let tail = diagram.tail;
        diagram.set_vertex_tau(tail, self.tau_ext());
        for (left, right) in self.arcs() {
            diagram.insert_arc(
                self.tau[left],
                self.tau[right],
                table.point(self.q[left] as usize),
                self.branch[left] as usize,
            );
        }
        diagram.set_cached_weight(self.log_modulus, self.phase);
        diagram
    }
}
