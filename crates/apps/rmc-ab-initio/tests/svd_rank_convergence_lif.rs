//! Observable-convergence gate for the SVD compression, sweeping the knob
//! production actually turns: the global rank budget of
//! [`EphSvd::truncate_global`], measured against the untruncated reference.
//!
//! Measured LiF picture (journal.md, "Plan B gate recalibration", 2026-09-03):
//! the global-budget error at k=Γ is NON-monotonic with ~±10 meV swings up to
//! total ~600 (σ-ordering is a poor importance ordering for a single k-row),
//! reaching ~1.4 meV at 800 of 1600. Total 300 carries ~7 meV systematic —
//! acceptable for the ±30 meV 20³ checkpoint, not for the final ~10 meV
//! extrapolation, which needs a budget ≥800 (or per-pair capping). Thresholds
//! below are those measurements with ~2× headroom.

use rmc_ab_initio::eph_svd::{EphSvd, DEFAULT_TOL, MAX_RANK};
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::vertex_table::{e2, VertexTable, DEFAULT_PHFREQ_CUTOFF_EV};
use std::path::PathBuf;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../datasets/LiF")
        .join(rel)
}

#[test]
fn second_order_energy_converges_with_total_rank() {
    let path = lif("out/LiF.epmatwp");
    if !path.exists() {
        eprintln!("skipping: epmatwp absent (regenerate via run_pipeline.nu)");
        return;
    }
    let epw = EpwData::read(lif("epwdata.fmt")).unwrap();
    let crystal = Crystal::read(lif("crystal.fmt")).unwrap();
    let wigner = Wigner::read(lif("wigner.fmt")).unwrap();
    let epmat = EpmatWp::read(&path, epw.nbnd, epw.nrr_k, epw.nmodes, epw.nrr_g).unwrap();
    if !epmat.data[0].is_finite() {
        eprintln!("skipping: epmatwp is non-finite");
        return;
    }

    // One decomposition; each budget copies only its kept components and
    // builds a small 8³ table — the exact objects production runs use. The
    // untruncated table is the reference the budgets must converge to.
    let full = EphSvd::compress(&epmat, DEFAULT_TOL, MAX_RANK);
    let reference_table =
        VertexTable::build(&epw, &crystal, &wigner, &full, 8, DEFAULT_PHFREQ_CUTOFF_EV).unwrap();
    let reference = e2(&reference_table, true, &usize::MAX);
    let reference_short = e2(&reference_table, false, &usize::MAX);
    eprintln!(
        "reference (untruncated, rank {}): E2_full {reference:+.9}  E2_short {reference_short:+.9}",
        full.rank()
    );

    let totals = [50, 100, 150, 200, 300, 400, 600, 800];
    let values = totals.map(|total| {
        let svd = full.truncate_global(total);
        let table =
            VertexTable::build(&epw, &crystal, &wigner, &svd, 8, DEFAULT_PHFREQ_CUTOFF_EV).unwrap();
        eprintln!("total {total}: pair_ranks {:?}", svd.pair_ranks());
        (
            e2(&table, true, &usize::MAX),
            e2(&table, false, &usize::MAX),
        )
    });

    eprintln!("total  E2_full[eV]  E2_short[eV]  err_vs_ref[eV]");
    for i in 0..totals.len() {
        eprintln!(
            "{:>5}  {:+.9}  {:+.9}  {:+.9}",
            totals[i],
            values[i].0,
            values[i].1,
            values[i].0 - reference,
        );
    }

    // Fine budget (production extrapolation quality): measured 1.37 meV.
    let err_800 = (values[7].0 - reference).abs();
    assert!(
        err_800 < 0.003,
        "total rank 800 is {err_800:.6} eV from the untruncated reference"
    );
    // Coarse budget (20³-checkpoint quality): measured 6.6 meV.
    let err_300 = (values[4].0 - reference).abs();
    assert!(
        err_300 < 0.012,
        "total rank 300 is {err_300:.6} eV from the untruncated reference"
    );
    // Long-range dominance: the exact polar term carries ~80% of E², which is
    // why modest short-range budgets are usable at all. Measured 20.8%.
    let short_fraction = reference_short.abs() / reference.abs();
    assert!(
        short_fraction < 0.35,
        "short-range E² fraction is {short_fraction:.3}"
    );
}
