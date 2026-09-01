//! `rmc_diagmc::Model` backed by exact uniform-grid vertex tables.

use std::sync::Arc;

use crate::vertex_table::VertexTable;
use nalgebra::Vector3;
use num_complex::Complex64;
use rand::{Rng, RngCore};
use rmc_diagmc::Model;

#[derive(Clone, Debug)]
pub struct AbInitioModel {
    pub table: Arc<VertexTable>,
    rank_cap: usize,
    q_weight: Vec<f64>,
    q_cdf: Vec<f64>,
    q_weight_sum: f64,
}

impl AbInitioModel {
    pub fn new(table: Arc<VertexTable>) -> Self {
        Self::with_rank(table, usize::MAX)
    }

    pub fn with_rank(table: Arc<VertexTable>, rank_cap: usize) -> Self {
        assert!(
            table.grid_n > 1,
            "the phonon grid must contain non-Γ points"
        );
        let mut q_weight = vec![0.0; table.len()];
        for (index, weight) in q_weight.iter_mut().enumerate().skip(1) {
            let start = index * table.nmodes;
            *weight = 1.0
                + table.long_ev[start..start + table.nmodes]
                    .iter()
                    .map(|g| g.norm_sqr())
                    .sum::<f64>();
        }
        let q_weight_sum: f64 = q_weight.iter().sum();
        let mut sum = 0.0;
        let q_cdf = q_weight
            .iter()
            .map(|&w| {
                sum += w;
                sum
            })
            .collect();
        Self {
            table,
            rank_cap,
            q_weight,
            q_cdf,
            q_weight_sum,
        }
    }
}

impl Model for AbInitioModel {
    fn n_branches(&self) -> usize {
        self.table.nmodes
    }

    fn electron_energy(&self, k: Vector3<f64>) -> f64 {
        self.table.electron_ev[self
            .table
            .index_of(k)
            .expect("electron momentum is not on the ab-initio grid")]
    }

    fn phonon_energy(&self, q: Vector3<f64>, nu: usize) -> f64 {
        assert!(nu < self.table.nmodes, "phonon branch is out of range");
        let q = self
            .table
            .index_of(q)
            .expect("phonon momentum is not on the ab-initio grid");
        self.table.phonon_ev[q * self.table.nmodes + nu]
    }

    fn vertex(&self, k: Vector3<f64>, q: Vector3<f64>, nu: usize) -> Complex64 {
        assert!(nu < self.table.nmodes, "phonon branch is out of range");
        let ki = self.table.index_of(k).expect("k off grid");
        let qi = self.table.index_of(q).expect("k off grid");
        self.table.full_indices_capped(ki, qi, nu, &self.rank_cap)
    }

    fn propose_phonon(&self, rng: &mut dyn RngCore) -> (Vector3<f64>, usize, f64) {
        let target = rng.gen::<f64>() * self.q_weight_sum;
        let q = self
            .q_cdf
            .partition_point(|&x| x <= target)
            .min(self.table.len() - 1);
        debug_assert_ne!(q, 0);
        let nu = rng.gen_range(0..self.table.nmodes);
        let point = self.table.point(q);
        (point, nu, self.propose_phonon_pdf(point, nu))
    }

    fn propose_phonon_pdf(&self, q: Vector3<f64>, nu: usize) -> f64 {
        if nu >= self.table.nmodes {
            return 0.0;
        }
        let Some(q) = self.table.index_of(q) else {
            return 0.0;
        };
        self.q_weight[q] / (self.q_weight_sum * self.table.nmodes as f64)
    }

    fn log_momentum_measure(&self) -> f64 {
        (self.table.len() as f64).ln()
    }

    fn phonon_frequency_cutoff(&self) -> f64 {
        self.table.phfreq_cutoff_ev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn model() -> AbInitioModel {
        let n = 3_usize;
        let nk = n.pow(3);
        AbInitioModel::new(Arc::new(VertexTable {
            grid_n: n,
            nbnd: 1,
            nmodes: 2,
            pair_offsets: vec![0, 0],
            s: vec![],
            utab: vec![],
            vtab: vec![],
            cbm_evec: vec![Complex64::new(1.0, 0.0); nk],
            electron_ev: vec![0.0; nk],
            phonon_ev: vec![1.0; nk * 2],
            long_ev: vec![Complex64::new(0.0, 0.0); nk * 2],
            phfreq_cutoff_ev: crate::vertex_table::DEFAULT_PHFREQ_CUTOFF_EV,
        }))
    }

    #[test]
    fn discrete_pdf_normalizes_and_excludes_gamma() {
        let model = model();
        let mut sum = 0.0;
        for q in 0..model.table.len() {
            for nu in 0..model.n_branches() {
                sum += model.propose_phonon_pdf(model.table.point(q), nu);
            }
        }
        assert!((sum - 1.0).abs() < 1e-12);
        assert_eq!(model.propose_phonon_pdf(Vector3::zeros(), 0), 0.0);
    }

    #[test]
    fn proposal_reports_exact_pdf() {
        let model = model();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..100 {
            let (q, nu, pdf) = model.propose_phonon(&mut rng);
            assert_ne!(q, Vector3::zeros());
            assert_eq!(pdf, model.propose_phonon_pdf(q, nu));
        }
    }

    #[test]
    #[should_panic(expected = "not on the ab-initio grid")]
    fn off_grid_energy_panics() {
        model().electron_energy(Vector3::new(0.1, 0.0, 0.0));
    }
}
