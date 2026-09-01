use std::sync::Arc;

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rmc_core::dispatch_update;
use rmc_core::mc::{Update, WeightedUpdate, WeightedUpdateSet};
use rmc_core::Result;

use crate::diagram::Diagram;
use crate::model::Model;
use crate::weight;

#[derive(Clone)]
pub struct AddPhonon {
    model: Arc<dyn Model>,
    proposal: Option<Diagram>,
    phase_change: Complex64,
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

        let mut phonon_rng = rand::rngs::StdRng::seed_from_u64(rng.gen());
        let (q, branch, reported_pdf) = self.model.propose_phonon(&mut phonon_rng);
        let pdf = self.model.propose_phonon_pdf(q, branch);
        if branch >= self.model.n_branches()
            || !pdf.is_finite()
            || pdf <= 0.0
            || (reported_pdf - pdf).abs() > 1.0e-10 * pdf.max(1.0)
        {
            return -1.0;
        }

        let mut tau1 = rng.gen::<f64>() * diagram.tau();
        let mut tau2 = rng.gen::<f64>() * diagram.tau();
        if tau1 > tau2 {
            std::mem::swap(&mut tau1, &mut tau2);
        }
        if tau1 <= 0.0 || tau2 <= tau1 || tau2 >= diagram.tau() {
            return -1.0;
        }

        let mut proposal = diagram.clone();
        proposal.insert_arc(tau1, tau2, q, branch);
        let ratio = weight::add_phonon_ratio(diagram, &proposal, self.model.as_ref());
        self.phase_change = ratio.phase;
        self.proposal = Some(proposal);
        ratio.modulus * diagram.tau().powi(2) / (2.0 * (diagram.order + 1) as f64 * pdf)
    }

    fn accept(&mut self, diagram: &mut Diagram) {
        *diagram = self.proposal.take().expect("accepted move has a proposal");
    }
}

#[derive(Clone)]
pub struct RemovePhonon {
    model: Arc<dyn Model>,
    proposal: Option<Diagram>,
    phase_change: Complex64,
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
        let arcs = diagram.arc_keys();
        let (left, right) = arcs[rng.gen_range(0..arcs.len())];
        let vertex = diagram.v(left);
        let pdf = self.model.propose_phonon_pdf(vertex.q, vertex.branch);
        if !pdf.is_finite() || pdf <= 0.0 {
            return -1.0;
        }
        let mut proposal = diagram.clone();
        proposal.remove_arc(left, right);
        let ratio = weight::remove_phonon_ratio(diagram, &proposal, self.model.as_ref());
        self.phase_change = ratio.phase;
        self.proposal = Some(proposal);
        ratio.modulus * 2.0 * diagram.order as f64 * pdf / diagram.tau().powi(2)
    }

    fn accept(&mut self, diagram: &mut Diagram) {
        *diagram = self.proposal.take().expect("accepted move has a proposal");
    }
}

#[derive(Clone)]
pub struct ChangeTau {
    model: Arc<dyn Model>,
    proposal: Option<Diagram>,
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
        let mut proposal = diagram.clone();
        proposal.set_vertex_tau(proposal.tail, tau);
        let ratio = weight::change_tau_ratio(diagram, &proposal, self.model.as_ref());
        self.phase_change = ratio.phase;
        self.proposal = Some(proposal);
        ratio.modulus
    }

    fn accept(&mut self, diagram: &mut Diagram) {
        *diagram = self.proposal.take().expect("accepted move has a proposal");
    }
}

#[derive(Clone)]
pub struct ChangeInternalTau {
    model: Arc<dyn Model>,
    proposal: Option<Diagram>,
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
        let mut proposal = diagram.clone();
        proposal.set_vertex_tau(key, tau);
        let ratio = weight::change_internal_tau_ratio(diagram, &proposal, self.model.as_ref());
        self.phase_change = ratio.phase;
        self.proposal = Some(proposal);
        ratio.modulus
    }

    fn accept(&mut self, diagram: &mut Diagram) {
        *diagram = self.proposal.take().expect("accepted move has a proposal");
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
