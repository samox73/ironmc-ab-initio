//! Probe B: all-orders DMC ground-state E0 vs SVD rank (n_sv).
//!
//! Build the vertex table ONCE at max rank, then sweep `rank_cap` on the model
//! (cheap — capping the rank loop in `full_indices_capped`) and run the DiagMC
//! engine at each. Plateau of E0(n_sv) within MC error = the production rank.
//!
//! Fixed-τ thermodynamic estimates make `mu` irrelevant. The 100k-step warm-up
//! is required because the initial order-zero diagram is atypical at τ=60 eV⁻¹.

use std::path::PathBuf;
use std::sync::Arc;

use rmc_ab_initio::{
    eph_svd::EphSvd,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let Some((e, c, w, wp)) = load() else {
        panic!("could not load data");
    };

    let grid_n = 8;
    // Ascending: rank 10 is the fast scout (read E0/sign/order + tune mu here),
    // higher ranks get progressively slower (vertex cost is ~linear in rank).
    let ranks = [100, 150, 200];

    let svd = time!(EphSvd::compress; &wp, 1e-8, *ranks.iter().max().unwrap());
    let table =
        Arc::new(time!(VertexTable::build; &e, &c, &w, &svd, grid_n, DEFAULT_PHFREQ_CUTOFF_EV)?);

    let max_tau = 60.0;
    // Optional CLI arg: steps per chain (default 100M). Use a small value to
    // scout throughput before committing to the full run.
    let max_steps: u64 = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("max_steps must be an integer"))
        .unwrap_or(100_000_000);

    println!("rank_cap  E0[eV]            sign    mean_order");
    for rank_cap in ranks {
        let model = Arc::new(AbInitioModel::with_rank(table.clone(), rank_cap));
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
            "{rank_cap:>8}  {:+.4} ± {:.4}  {:.3}   {:.2}",
            r.e0_thermo.mean, r.e0_thermo.stderr, r.sign.mean, r.mean_order.mean
        );
    }
    Ok(())
}
