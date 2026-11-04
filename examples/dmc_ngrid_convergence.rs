//! Fixed-τ grid ladder for the LiF electron-polaron binding energy: one
//! globally-truncated SVD (production budget), one table + DMC run per grid
//! size.
//!
//! NOTE on rank semantics: since the Plan-B rework, `total_rank = 150` is a
//! GLOBAL budget across all band pairs (`compress_global`), not the per-pair
//! `rank_cap` the pre-2026-09-02 journal entries used — E0 at "rank 150" is
//! not comparable across that change.

use std::path::PathBuf;
use std::sync::Arc;

use rmc_ab_initio::{
    eph_svd::EphSvd,
    epw::{Crystal, EpmatWp, EpwData, Wigner},
    model_abinitio::AbInitioModel,
    vertex_table::{DEFAULT_MAX_BYTES, DEFAULT_PHFREQ_CUTOFF_EV, TablePrecision, VertexTable},
};
use rmc_core::time;
use rmc_diagmc::{DiagConfig, run};

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/LiF")
        .join(rel)
}

fn load() -> Option<(EpwData, Crystal, Wigner, EpmatWp)> {
    let wp_path = lif("out/LiF.epmatwp");
    if !wp_path.exists() {
        eprintln!("skipping: epmatwp absent (regenerate via run_pipeline.nu)");
        return None;
    }
    let e = EpwData::read(lif("epwdata.fmt")).unwrap();
    let c = Crystal::read(lif("crystal.fmt")).unwrap();
    let w = Wigner::read(lif("wigner.fmt")).unwrap();
    let wp = EpmatWp::read(&wp_path, e.nbnd, e.nrr_k, e.nmodes, e.nrr_g).unwrap();
    if !wp.data[0].is_finite() {
        eprintln!("skipping: epmatwp is all-NaN; regenerate the EPW e-ph tensor");
        return None;
    }
    Some((e, c, w, wp))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let Some((e, c, w, wp)) = load() else {
        panic!("could not load data");
    };

    let grid_ns = [8, 12, 16, 20];
    let total_rank = 150;
    let svd = time!(EphSvd::compress_global; &wp, 1e-8, total_rank);

    println!("grid_n  E0[eV]            sign    mean_order");
    for grid_n in grid_ns {
        let table = Arc::new(
            time!(VertexTable::build; &e, &c, &w, &svd, grid_n, DEFAULT_PHFREQ_CUTOFF_EV, TablePrecision::F32, DEFAULT_MAX_BYTES)?,
        );
        let k0 = table.index_of(nalgebra::Vector3::zeros()).unwrap();
        let (mut max_g, mut arg_g) = (0.0f64, (0, 0));
        let (mut min_w, mut arg_w) = (f64::MAX, (0, 0));
        for q in 1..table.len() {
            for nu in 0..table.nmodes {
                let w = table.phonon_ev[q * table.nmodes + nu].abs();
                if w < min_w {
                    min_w = w;
                    arg_w = (q, nu);
                }
                let g = table.full_indices(k0, q, nu).norm();
                if g > max_g {
                    max_g = g;
                    arg_g = (q, nu);
                }
            }
        }
        println!(
            "grid {grid_n}: min|w|={min_w:.3e} eV @{arg_w:?}   max|g(G,q,nu)|={max_g:.3e} eV @{arg_g:?}"
        );

        let max_tau = 60.0;

        let model = Arc::new(AbInitioModel::new(table.clone()));
        let cfg = DiagConfig {
            max_tau,
            tau_fit: max_tau / 2.0,
            fixed_tau: true,
            max_order: 500,
            chains: 10,
            max_steps: 100_000_000,
            warmup_steps: 100_000,
            ..DiagConfig::default()
        };
        let r = time!(run; model, cfg)?;
        println!(
            "{grid_n:>6}  {:+.4} ± {:.4}  {:.3}   {:.2}",
            r.e0_thermo.mean, r.e0_thermo.stderr, r.sign.mean, r.mean_order.mean
        );
    }
    Ok(())
}
