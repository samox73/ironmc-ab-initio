//! Shared test fixtures.

use std::sync::Arc;

use num_complex::{Complex, Complex64};
use rand::{Rng, SeedableRng};
use rmc_ab_initio::model_abinitio::AbInitioModel;
use rmc_ab_initio::vertex_table::{Tab, VertexTable, DEFAULT_PHFREQ_CUTOFF_EV};

/// Small physical-ish random table: grid 4³, nbnd 2, nmodes 2, rank 8,
/// nonzero long-range part (non-uniform q proposal weights).
pub fn synthetic_model(seed: u64) -> Arc<AbInitioModel> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let grid_n = 4_usize;
    let nk = grid_n.pow(3);
    let nbnd = 2_usize;
    let nmodes = 2_usize;
    let ncomp = 8_usize;
    let mut c32 =
        |scale: f32| Complex::new(rng.gen_range(-scale..scale), rng.gen_range(-scale..scale));
    let wtab: Vec<Complex<f32>> = (0..nk * ncomp).map(|_| c32(0.1)).collect();
    let vtab: Vec<Complex<f32>> = (0..nk * nmodes * ncomp).map(|_| c32(0.1)).collect();
    let cbm_evec: Vec<Complex64> = (0..nk * nbnd)
        .map(|_| Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
        .collect();
    let electron_ev: Vec<f64> = (0..nk).map(|_| rng.gen_range(0.0..2.0)).collect();
    let phonon_ev: Vec<f64> = (0..nk * nmodes).map(|_| rng.gen_range(0.02..0.1)).collect();
    let long_ev: Vec<Complex64> = (0..nk * nmodes)
        .map(|_| Complex64::new(rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)))
        .collect();
    Arc::new(AbInitioModel::new(Arc::new(VertexTable {
        grid_n,
        nbnd,
        nmodes,
        pair_offsets: vec![0, 2, 4, 6, 8],
        s: vec![1.0; ncomp],
        wtab: Tab::F32(wtab),
        bra_band: vec![0, 0, 0, 0, 1, 1, 1, 1],
        vtab: Tab::F32(vtab),
        cbm_evec,
        electron_ev,
        phonon_ev,
        long_ev,
        phfreq_cutoff_ev: DEFAULT_PHFREQ_CUTOFF_EV,
    })))
}
