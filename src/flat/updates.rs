//! The four `rmc-diagmc` updates on the flat diagram, drawn through
//! GPU-portable samplers (every draw is a plain u01; `gen_range`'s rejection
//! sampling is replaced by `uniform_index_from_u01`, so a Philox word stream
//! maps 1:1 onto the kernel). Ratio math and guards mirror
//! `rmc_diagmc::updates` exactly — the parity suite pins per-update agreement
//! at 1e-10 relative. Proposal fields are `pub` for those tests.

use std::sync::Arc;

use rand::{Rng, RngExt};
use rmc_core::dispatch_update;
use rmc_core::mc::{WeightedUpdate, WeightedUpdateSet};
use rmc_core::random::{exponential_sample_bounded, uniform_index_from_u01};
use rmc_core::Result;

use super::{weight, wrap_sub, FlatDiagram};
use crate::model_abinitio::AbInitioModel;
use rmc_diagmc::updates::{time_proposal_pdf, DELTA_TAU_LIMIT};
use rmc_diagmc::weight::LocalRatio;

/// λ for the truncated-exponential τ₂ proposal, index-level mirror of
/// `rmc_diagmc::updates::decay_rate` (same 1e-6 floor).
fn decay_rate(model: &AbInitioModel, grid_n: u32, p: u32, q: u32, nu: u32) -> f64 {
    (model.phonon_ev_index(q as usize, nu as usize)
        + model.electron_ev_index(wrap_sub(grid_n, p, q) as usize)
        - model.electron_ev_index(p as usize))
    .max(1.0e-6)
}

#[derive(Clone)]
pub struct AddPhonon {
    pub model: Arc<AbInitioModel>,
    pub tau1: f64,
    pub tau2: f64,
    pub q: u32,
    pub branch: u32,
    pub ratio: LocalRatio,
}

impl AddPhonon {
    pub fn new(model: Arc<AbInitioModel>) -> Self {
        Self {
            model,
            tau1: 0.0,
            tau2: 0.0,
            q: 0,
            branch: 0,
            ratio: unit_ratio(),
        }
    }

    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &mut FlatDiagram, rng: &mut R) -> f64 {
        if d.order >= d.max_order || d.tau_ext() <= 2.0 * f64::EPSILON {
            return -1.0;
        }
        let (q, branch, q_pdf) = self.model.propose_phonon_index(rng.random(), rng.random());
        let (q, branch) = (q as u32, branch as u32);
        if !q_pdf.is_finite() || q_pdf <= 0.0 {
            return -1.0;
        }

        let tau1 = rng.random::<f64>() * d.tau_ext();
        let segment1 = d.segment_at(tau1);
        if tau1 - d.tau[segment1] < DELTA_TAU_LIMIT || d.tau[segment1 + 1] - tau1 < DELTA_TAU_LIMIT
        {
            return -1.0;
        }
        let p = d.p_out[segment1];
        let lambda = decay_rate(&self.model, d.grid_n, p, q, branch);
        let tau2 = exponential_sample_bounded(rng.random(), lambda, tau1, d.tau_ext());
        if tau2 - tau1 < DELTA_TAU_LIMIT {
            return -1.0;
        }
        let segment2 = d.segment_at(tau2);
        if tau2 - d.tau[segment2] < DELTA_TAU_LIMIT || d.tau[segment2 + 1] - tau2 < DELTA_TAU_LIMIT
        {
            return -1.0;
        }
        let tau_pdf = time_proposal_pdf(d.tau_ext(), tau1, tau2, lambda);

        let ratio = weight::insert_arc_ratio(d, &self.model, tau1, tau2, q, branch);
        self.tau1 = tau1;
        self.tau2 = tau2;
        self.q = q;
        self.branch = branch;
        self.ratio = ratio;
        ratio.modulus() / ((d.order + 1) as f64 * q_pdf * tau_pdf)
    }

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        d.insert_arc(self.tau1, self.tau2, self.q, self.branch);
        d.apply_ratio(&self.ratio);
    }
}

#[derive(Clone)]
pub struct RemovePhonon {
    pub model: Arc<AbInitioModel>,
    pub left: usize,
    pub right: usize,
    pub ratio: LocalRatio,
}

impl RemovePhonon {
    pub fn new(model: Arc<AbInitioModel>) -> Self {
        Self {
            model,
            left: 0,
            right: 0,
            ratio: unit_ratio(),
        }
    }

    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &mut FlatDiagram, rng: &mut R) -> f64 {
        if d.order <= d.min_order {
            return -1.0;
        }
        let nth = uniform_index_from_u01(rng.random(), d.order);
        let (left, right) = d.nth_arc(nth).expect("selected arc must exist");
        let q = d.q[left];
        let branch = d.branch[left];
        let q_pdf = self.model.phonon_pdf_index(q as usize);
        if !q_pdf.is_finite() || q_pdf <= 0.0 {
            return -1.0;
        }
        let lambda = decay_rate(&self.model, d.grid_n, d.incoming_momentum(left), q, branch);
        let tau_pdf = time_proposal_pdf(d.tau_ext(), d.tau[left], d.tau[right], lambda);
        let ratio = weight::remove_arc_ratio(d, &self.model, left, right);
        self.left = left;
        self.right = right;
        self.ratio = ratio;
        ratio.modulus() * d.order as f64 * q_pdf * tau_pdf
    }

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        d.remove_arc(self.left, self.right);
        d.apply_ratio(&self.ratio);
    }
}

#[derive(Clone)]
pub struct ChangeTau {
    pub model: Arc<AbInitioModel>,
    pub tau: f64,
    pub ratio: LocalRatio,
}

impl ChangeTau {
    pub fn new(model: Arc<AbInitioModel>) -> Self {
        Self {
            model,
            tau: 0.0,
            ratio: unit_ratio(),
        }
    }

    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &mut FlatDiagram, rng: &mut R) -> f64 {
        let lower_tau = d.tau[d.tail() - 1];
        if lower_tau >= d.max_tau {
            return -1.0;
        }
        let tau = lower_tau + rng.random::<f64>() * (d.max_tau - lower_tau);
        if tau - lower_tau < DELTA_TAU_LIMIT {
            return -1.0;
        }
        let ratio = weight::shift_tail_ratio(d, &self.model, tau);
        self.tau = tau;
        self.ratio = ratio;
        ratio.modulus()
    }

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        let tail = d.tail();
        d.set_vertex_tau(tail, self.tau);
        d.apply_ratio(&self.ratio);
    }
}

#[derive(Clone)]
pub struct ChangeInternalTau {
    pub model: Arc<AbInitioModel>,
    pub key: usize,
    pub tau: f64,
    pub ratio: LocalRatio,
}

impl ChangeInternalTau {
    pub fn new(model: Arc<AbInitioModel>) -> Self {
        Self {
            model,
            key: 0,
            tau: 0.0,
            ratio: unit_ratio(),
        }
    }

    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &mut FlatDiagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            return -1.0;
        }
        // Positional keys 1..tail, exactly `gen_range(1..diagram.tail)`.
        let key = 1 + uniform_index_from_u01(rng.random(), 2 * d.order);
        let lower = d.tau[key - 1];
        let upper = d.tau[key + 1];
        let tau = lower + rng.random::<f64>() * (upper - lower);
        if tau - lower < DELTA_TAU_LIMIT || upper - tau < DELTA_TAU_LIMIT {
            return -1.0;
        }
        let ratio = weight::shift_vertex_tau_ratio(d, &self.model, key, tau);
        self.key = key;
        self.tau = tau;
        self.ratio = ratio;
        ratio.modulus()
    }

    pub fn accept(&mut self, d: &mut FlatDiagram) {
        d.set_vertex_tau(self.key, self.tau);
        d.apply_ratio(&self.ratio);
    }
}

fn unit_ratio() -> LocalRatio {
    LocalRatio {
        log_modulus: 0.0,
        phase: num_complex::Complex64::new(1.0, 0.0),
    }
}

dispatch_update! {
    #[derive(Clone)]
    pub enum FlatAbUpdate<FlatDiagram> {
        ChangeTau(ChangeTau),
        ChangeInternalTau(ChangeInternalTau),
        AddPhonon(AddPhonon),
        RemovePhonon(RemovePhonon),
    }
}

/// The four-move set (free external τ). Order matches
/// `rmc_diagmc::updates::update_set`, and the device selection
/// indexes into this same order.
pub fn update_set(model: Arc<AbInitioModel>) -> Result<WeightedUpdateSet<FlatAbUpdate>> {
    WeightedUpdateSet::new(vec![
        WeightedUpdate::new(FlatAbUpdate::ChangeTau(ChangeTau::new(model.clone())), 1.0),
        WeightedUpdate::new(
            FlatAbUpdate::ChangeInternalTau(ChangeInternalTau::new(model.clone())),
            1.0,
        ),
        WeightedUpdate::new(FlatAbUpdate::AddPhonon(AddPhonon::new(model.clone())), 1.0),
        WeightedUpdate::new(FlatAbUpdate::RemovePhonon(RemovePhonon::new(model)), 1.0),
    ])
}

/// The three-move set for `fixed_tau` runs (production LiF configs).
pub fn update_set_fixed_tau(model: Arc<AbInitioModel>) -> Result<WeightedUpdateSet<FlatAbUpdate>> {
    WeightedUpdateSet::new(vec![
        WeightedUpdate::new(
            FlatAbUpdate::ChangeInternalTau(ChangeInternalTau::new(model.clone())),
            1.0,
        ),
        WeightedUpdate::new(FlatAbUpdate::AddPhonon(AddPhonon::new(model.clone())), 1.0),
        WeightedUpdate::new(FlatAbUpdate::RemovePhonon(RemovePhonon::new(model)), 1.0),
    ])
}
