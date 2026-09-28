//! Phase A1 correctness: all three vertex-dot kernel variants match
//! `VertexTable::short_indices_capped` elementwise at 1e-6 relative on the
//! CubeCL CPU runtime, on a synthetic table with ragged band boundaries.

#![cfg(feature = "gpu-cpu")]

use num_complex::{Complex, Complex64};
use rand::{Rng, SeedableRng};
use rmc_ab_initio::gpu::vertex_bench::{max_rel_error, random_evals, DeviceTables, Variant};
use rmc_ab_initio::vertex_table::{Tab, VertexTable, DEFAULT_PHFREQ_CUTOFF_EV};

fn synthetic_table() -> VertexTable {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xab1);
    let grid_n = 3_usize;
    let nk = grid_n.pow(3);
    let nbnd = 4_usize;
    let nmodes = 2_usize;
    // 16 ragged pairs, total rank 24 (multiple of the vector width), band
    // boundaries at 7/16/21 — none a multiple of 4, exercising the vector
    // kernel's scalar head/tail paths.
    let pair_offsets: Vec<usize> = vec![0, 2, 3, 5, 7, 8, 10, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24];
    let ncomp = *pair_offsets.last().unwrap();
    let mut bra_band = vec![0u8; ncomp];
    for n in 0..nbnd {
        for m in 0..nbnd {
            let pair = n * nbnd + m;
            for g in pair_offsets[pair]..pair_offsets[pair + 1] {
                bra_band[g] = n as u8;
            }
        }
    }
    let mut cf32 = |len: usize| -> Tab {
        Tab::F32(
            (0..len)
                .map(|_| Complex::new(rng.gen_range(-1.0f32..1.0), rng.gen_range(-1.0f32..1.0)))
                .collect(),
        )
    };
    let wtab = cf32(nk * ncomp);
    let vtab = cf32(nk * nmodes * ncomp);
    let cbm_evec: Vec<Complex64> = (0..nk * nbnd)
        .map(|_| Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
        .collect();
    VertexTable {
        grid_n,
        nbnd,
        nmodes,
        pair_offsets,
        s: vec![1.0; ncomp],
        wtab,
        bra_band,
        vtab,
        cbm_evec,
        electron_ev: vec![0.0; nk],
        phonon_ev: vec![1.0; nk * nmodes],
        long_ev: vec![Complex64::new(0.0, 0.0); nk * nmodes],
        phfreq_cutoff_ev: DEFAULT_PHFREQ_CUTOFF_EV,
    }
}

#[test]
fn all_variants_match_cpu_oracle() {
    let table = synthetic_table();
    let client = <cubecl::cpu::CpuRuntime as cubecl::prelude::Runtime>::client(&Default::default());
    let tables = DeviceTables::upload(client, &table);
    let (ks, qs, nus) = random_evals(&table, 200, 42);
    for variant in Variant::ALL {
        let (values, _) = tables.evaluate(variant, &ks, &qs, &nus, false);
        let err = max_rel_error(&table, &ks, &qs, &nus, &values);
        assert!(
            err <= 1.0e-6,
            "variant {} exceeds tolerance: {err:.3e}",
            variant.name()
        );
    }
}
