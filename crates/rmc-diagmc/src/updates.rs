use std::sync::Arc;

use nalgebra::Vector3;
use num_complex::Complex64;
use rand::{Error, Rng, RngCore};
use rmc_core::dispatch_update;
use rmc_core::mc::{Update, WeightedUpdate, WeightedUpdateSet};
use rmc_core::random::{exponential_pdf_bounded, exponential_sample_bounded};
use rmc_core::Result;

use crate::diagram::Diagram;
use crate::model::Model;
use crate::weight;

#[derive(Clone)]
pub struct AddPhonon {
    model: Arc<dyn Model>,
    proposal: Option<AddProposal>,
    phase_change: Complex64,
}

#[derive(Clone, Copy)]
struct AddProposal {
    tau1: f64,
    tau2: f64,
    q: Vector3<f64>,
    branch: usize,
    ratio: weight::LocalRatio,
}

impl AddPhonon {
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            proposal: None,
            phase_change: Complex64::new(1.0, 0.0),
        }
    }

    pub fn phase_change(&self) -> Complex64 {
        self.phase_change
    }
}

impl Update<Diagram> for AddPhonon {
    fn attempt<R: Rng + ?Sized>(&mut self, diagram: &mut Diagram, rng: &mut R) -> f64 {
        self.proposal = None;
        if diagram.order >= diagram.max_order || diagram.tau() <= 2.0 * f64::EPSILON {
            return -1.0;
        }

        let (q, branch, q_pdf) = self.model.propose_phonon(&mut RngAdapter(rng));
        debug_assert!(branch < self.model.n_branches());
        debug_assert!(
            (q_pdf - self.model.propose_phonon_pdf(q, branch)).abs() <= 1.0e-10 * q_pdf.max(1.0)
        );
        if !q_pdf.is_finite() || q_pdf <= 0.0 {
            return -1.0;
        }

        let tau1 = rng.gen::<f64>() * diagram.tau();
        let p = diagram.v(weight::segment_at(diagram, tau1)).p_out;
        let lambda = decay_rate(self.model.as_ref(), p, q, branch);
        let tau2 = exponential_sample_bounded(rng.gen(), lambda, tau1, diagram.tau());
        if tau1 <= 0.0 || tau2 <= tau1 || tau2 >= diagram.tau() {
            return -1.0;
        }
        let tau_pdf = time_proposal_pdf(diagram.tau(), tau1, tau2, lambda);

        let ratio = weight::insert_arc_ratio(diagram, self.model.as_ref(), tau1, tau2, q, branch);
        self.phase_change = ratio.phase;
        self.proposal = Some(AddProposal {
            tau1,
            tau2,
            q,
            branch,
            ratio,
        });
        ratio.modulus() / ((diagram.order + 1) as f64 * q_pdf * tau_pdf)
    }

    fn accept(&mut self, diagram: &mut Diagram) {
        let proposal = self.proposal.take().expect("accepted move has a proposal");
        diagram.insert_arc(proposal.tau1, proposal.tau2, proposal.q, proposal.branch);
        diagram.apply_ratio(&proposal.ratio);
    }
}

#[derive(Clone)]
pub struct RemovePhonon {
    model: Arc<dyn Model>,
    proposal: Option<RemoveProposal>,
    phase_change: Complex64,
}

#[derive(Clone, Copy)]
struct RemoveProposal {
    left: usize,
    right: usize,
    ratio: weight::LocalRatio,
}

impl RemovePhonon {
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            proposal: None,
            phase_change: Complex64::new(1.0, 0.0),
        }
    }

    pub fn phase_change(&self) -> Complex64 {
        self.phase_change
    }
}

impl Update<Diagram> for RemovePhonon {
    fn attempt<R: Rng + ?Sized>(&mut self, diagram: &mut Diagram, rng: &mut R) -> f64 {
        self.proposal = None;
        if diagram.order <= diagram.min_order {
            return -1.0;
        }
        let (left, right) = diagram
            .nth_arc(rng.gen_range(0..diagram.arc_count()))
            .expect("selected arc must exist");
        let vertex = diagram.v(left);
        let q_pdf = self.model.propose_phonon_pdf(vertex.q, vertex.branch);
        if !q_pdf.is_finite() || q_pdf <= 0.0 {
            return -1.0;
        }
        let lambda = decay_rate(
            self.model.as_ref(),
            diagram.incoming_momentum(left),
            vertex.q,
            vertex.branch,
        );
        let tau_pdf = time_proposal_pdf(diagram.tau(), vertex.tau, diagram.v(right).tau, lambda);
        let ratio = weight::remove_arc_ratio(diagram, self.model.as_ref(), left, right);
        self.phase_change = ratio.phase;
        self.proposal = Some(RemoveProposal { left, right, ratio });
        ratio.modulus() * diagram.order as f64 * q_pdf * tau_pdf
    }

    fn accept(&mut self, diagram: &mut Diagram) {
        let proposal = self.proposal.take().expect("accepted move has a proposal");
        diagram.remove_arc(proposal.left, proposal.right);
        diagram.apply_ratio(&proposal.ratio);
    }
}

#[derive(Clone)]
pub struct ChangeTau {
    model: Arc<dyn Model>,
    proposal: Option<(f64, weight::LocalRatio)>,
    phase_change: Complex64,
}

impl ChangeTau {
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            proposal: None,
            phase_change: Complex64::new(1.0, 0.0),
        }
    }
}

impl Update<Diagram> for ChangeTau {
    fn attempt<R: Rng + ?Sized>(&mut self, diagram: &mut Diagram, rng: &mut R) -> f64 {
        self.proposal = None;
        let lower = diagram.v(diagram.tail).prev;
        let lower_tau = if lower == diagram.head { 0.0 } else { diagram.v(lower).tau };
        if lower_tau >= diagram.max_tau {
            return -1.0;
        }
        let tau = lower_tau + rng.gen::<f64>() * (diagram.max_tau - lower_tau);
        if tau <= lower_tau {
            return -1.0;
        }
        let ratio = weight::shift_tail_ratio(diagram, self.model.as_ref(), tau);
        self.phase_change = ratio.phase;
        self.proposal = Some((tau, ratio));
        ratio.modulus()
    }

    fn accept(&mut self, diagram: &mut Diagram) {
        let (tau, ratio) = self.proposal.take().expect("accepted move has a proposal");
        diagram.set_vertex_tau(diagram.tail, tau);
        diagram.apply_ratio(&ratio);
    }
}

#[derive(Clone)]
pub struct ChangeInternalTau {
    model: Arc<dyn Model>,
    proposal: Option<(usize, f64, weight::LocalRatio)>,
    phase_change: Complex64,
}

impl ChangeInternalTau {
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            proposal: None,
            phase_change: Complex64::new(1.0, 0.0),
        }
    }
}

impl Update<Diagram> for ChangeInternalTau {
    fn attempt<R: Rng + ?Sized>(&mut self, diagram: &mut Diagram, rng: &mut R) -> f64 {
        self.proposal = None;
        if diagram.order == 0 {
            return -1.0;
        }
        let key = rng.gen_range(1..diagram.tail);
        let lower = diagram.v(key - 1).tau;
        let upper = diagram.v(key + 1).tau;
        let tau = lower + rng.gen::<f64>() * (upper - lower);
        if tau <= lower || tau >= upper {
            return -1.0;
        }
        let ratio = weight::shift_vertex_tau_ratio(diagram, self.model.as_ref(), key, tau);
        self.phase_change = ratio.phase;
        self.proposal = Some((key, tau, ratio));
        ratio.modulus()
    }

    fn accept(&mut self, diagram: &mut Diagram) {
        let (key, tau, ratio) = self.proposal.take().expect("accepted move has a proposal");
        diagram.set_vertex_tau(key, tau);
        diagram.apply_ratio(&ratio);
    }
}

fn decay_rate(model: &dyn Model, p: Vector3<f64>, q: Vector3<f64>, branch: usize) -> f64 {
    (model.phonon_energy(q, branch) + model.electron_energy(p - q) - model.electron_energy(p))
        .max(1.0e-6)
}

/// Joint density of (τ1 uniform on (0, τ_ext), τ2 truncated-exponential on
/// (τ1, τ_ext)); the exponential part is rmc-core's shared sampler so the draw
/// in `AddPhonon` and this reverse-move density can never drift apart.
fn time_proposal_pdf(tau_ext: f64, tau1: f64, tau2: f64, lambda: f64) -> f64 {
    exponential_pdf_bounded(tau2, lambda, tau1, tau_ext) / tau_ext
}

struct RngAdapter<'a, R: ?Sized>(&'a mut R);

impl<R: RngCore + ?Sized> RngCore for RngAdapter<'_, R> {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> std::result::Result<(), Error> {
        self.0.try_fill_bytes(dest)
    }
}

dispatch_update! {
    #[derive(Clone)]
    pub enum DiagUpdate<Diagram> {
        ChangeTau(ChangeTau),
        ChangeInternalTau(ChangeInternalTau),
        AddPhonon(AddPhonon),
        RemovePhonon(RemovePhonon),
    }
}

pub fn update_set(model: Arc<dyn Model>) -> Result<WeightedUpdateSet<DiagUpdate>> {
    WeightedUpdateSet::new(vec![
        WeightedUpdate::new(DiagUpdate::ChangeTau(ChangeTau::new(model.clone())), 1.0),
        WeightedUpdate::new(
            DiagUpdate::ChangeInternalTau(ChangeInternalTau::new(model.clone())),
            1.0,
        ),
        WeightedUpdate::new(DiagUpdate::AddPhonon(AddPhonon::new(model.clone())), 1.0),
        WeightedUpdate::new(DiagUpdate::RemovePhonon(RemovePhonon::new(model)), 1.0),
    ])
}

pub fn update_set_fixed_tau(model: Arc<dyn Model>) -> Result<WeightedUpdateSet<DiagUpdate>> {
    WeightedUpdateSet::new(vec![
        WeightedUpdate::new(
            DiagUpdate::ChangeInternalTau(ChangeInternalTau::new(model.clone())),
            1.0,
        ),
        WeightedUpdate::new(DiagUpdate::AddPhonon(AddPhonon::new(model.clone())), 1.0),
        WeightedUpdate::new(DiagUpdate::RemovePhonon(RemovePhonon::new(model)), 1.0),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FrohlichModel;
    use crate::weight::weight_modulus;
    use nalgebra::Vector3;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn add_then_remove_is_detailed_balance_inverse() {
        let model: Arc<dyn Model> = Arc::new(FrohlichModel::new(1.0));
        let original = Diagram::new(-1.2, Vector3::zeros(), 10.0, 0, 10);
        let mut diagram = original.clone();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        let mut add = AddPhonon::new(model.clone());
        let add_probability = add.attempt(&mut diagram, &mut rng);
        assert!(add_probability.is_finite() && add_probability > 0.0);
        add.accept(&mut diagram);

        let mut remove = RemovePhonon::new(model.clone());
        let remove_probability = remove.attempt(&mut diagram, &mut rng);
        assert!(remove_probability.is_finite() && remove_probability > 0.0);
        remove.accept(&mut diagram);

        assert!((add_probability * remove_probability - 1.0).abs() < 1.0e-10);
        assert!(
            (weight_modulus(&diagram, model.as_ref()) - weight_modulus(&original, model.as_ref()))
                .abs()
                < 1.0e-10
        );
        assert_eq!(diagram.vertices, original.vertices);
        assert_eq!(diagram.order, original.order);
    }
}
