//! Probe B: all-orders DMC ground-state E0 vs total SVD rank.
//!
//! Decompose once at the per-pair cap, then globally truncate and build a table
//! for each production budget. Plateau of E0(total rank) within MC error = the
//! production rank.
//!
//! Fixed-τ thermodynamic estimates make `mu` irrelevant. The 100k-step warm-up
//! is required because the initial order-zero diagram is atypical at τ=60 eV⁻¹.

use std::path::PathBuf;
use std::sync::Arc;

use rmc_ab_initio::{
    eph_svd::{EphSvd, MAX_RANK},
    epw::{Crystal, EpmatWp, EpwData, Wigner},
    model_abinitio::AbInitioModel,
    vertex_table::{VertexTable, DEFAULT_PHFREQ_CUTOFF_EV},
};
use rmc_core::time;
use rmc_diagmc::{run, DiagConfig};

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../datasets/LiF")
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

pub struct RankResult {
    pub total_rank: usize,
    pub mean: f64,
    pub stderr: f64,
}

pub fn convergence(max_steps: u64) -> Result<Option<Vec<RankResult>>, Box<dyn std::error::Error>> {
    let Some((e, c, w, wp)) = load() else {
        return Ok(None);
    };

    let grid_n = 8;
    // Production budget 300 vs the untruncated reference (1600). Measured at
    // the E₀ level (journal 2026-09-03): Δ_rank(300) = 0.7 ± 0.5 meV at 8³ —
    // the Γ-row E² proxy (~7 meV) overestimates the DMC systematic, which
    // averages compression error over the whole BZ. Rank 150 is genuinely too
    // low (4.6 meV measured); 300 matches FEP-DMC's nsvd=20/pair ≈ 320 total.
    let ranks = [300, 1600];
    let svd = time!(EphSvd::compress; &wp, 1e-8, MAX_RANK);
    log::info!(
        "max SVD reconstruction error: {:.3e}",
        svd.max_relative_error(&wp)
    );

    let max_tau = 60.0;
    let mut results = Vec::with_capacity(ranks.len());
    println!("total_rank  E0[eV]            sign    mean_order");
    for total_rank in ranks {
        let truncated = svd.clone().truncate_global(total_rank);
        let table = Arc::new(
            time!(VertexTable::build; &e, &c, &w, &truncated, grid_n, DEFAULT_PHFREQ_CUTOFF_EV)?,
        );
        let model = Arc::new(AbInitioModel::new(table));
        let cfg = DiagConfig {
            max_tau,
            tau_fit: max_tau / 2.0,
            fixed_tau: true,
            max_order: 500,
            chains: 10,
            max_steps,
            warmup_steps: 100_000,
            ..DiagConfig::default()
        };
        let r = time!(run; model, cfg)?;
        println!(
            "{total_rank:>10}  {:+.4} ± {:.4}  {:.3}   {:.2}",
            r.e0_thermo.mean, r.e0_thermo.stderr, r.sign.mean, r.mean_order.mean
        );
        results.push(RankResult {
            total_rank,
            mean: r.e0_thermo.mean,
            stderr: r.e0_thermo.stderr,
        });
    }
    Ok(Some(results))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let max_steps = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("max_steps must be an integer"))
        .unwrap_or(100_000_000);
    if convergence(max_steps)?.is_none() {
        return Err("could not load data".into());
    }
    Ok(())
}
