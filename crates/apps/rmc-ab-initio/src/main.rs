use std::path::{Path, PathBuf};

use rmc_ab_initio::config::AbInitioConfig;
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::model::{regular_grid, ReducedModel};
use rmc_ab_initio::reconstruct::GammaModel;
use rmc_ab_initio::run::run_and_write;

/// Build the reduced `|g(Γ,q,LO)|²` artifact from an EPW dataset dir (holding
/// `epwdata.fmt`, `crystal.fmt`, `wigner.fmt`, `out/LiF.epmatwp`) on a regular
/// `n×n×n` q-grid. Needs the ~146 MB `epmatwp` present on disk.
fn build_reduced(dataset: &Path, n: usize) -> Result<ReducedModel, Box<dyn std::error::Error>> {
    let e = EpwData::read(dataset.join("epwdata.fmt"))?;
    let c = Crystal::read(dataset.join("crystal.fmt"))?;
    let w = Wigner::read(dataset.join("wigner.fmt"))?;
    let wp = EpmatWp::read(
        dataset.join("out/LiF.epmatwp"),
        e.nbnd,
        e.nrr_k,
        e.nmodes,
        e.nrr_g,
    )?;
    if !wp.data[0].is_finite() {
        return Err("epmatwp is all-NaN; regenerate the EPW e-ph tensor".into());
    }
    let gamma = GammaModel::build(&e, &c, &w, &wp);
    Ok(ReducedModel::build(&gamma, &e, &c, &w, &regular_grid(n)))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("def") => println!("{}", AbInitioConfig::default().to_json_string()?),
        Some("model") => {
            let dataset = args
                .next()
                .ok_or("usage: model <dataset_dir> <out.json> [grid_n]")?;
            let out = args
                .next()
                .ok_or("usage: model <dataset_dir> <out.json> [grid_n]")?;
            let n: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(8);
            let m = build_reduced(Path::new(&dataset), n)?;
            m.save(&out)?;
            println!(
                "wrote {out}: {} q-points, omega_LO={:.1} meV",
                m.q.len(),
                m.omega_lo_mev
            );
        }
        Some("alpha") => {
            let cfg = match args.next() {
                Some(path) => AbInitioConfig::load_json(path)?,
                None => AbInitioConfig::default(),
            };
            println!("alpha_LiF: {:.4}", cfg.material.frohlich_alpha());
        }
        Some(path) => {
            let cfg = AbInitioConfig::load_json(path)?;
            let dir = args
                .next()
                .map_or_else(|| PathBuf::from("results-abinitio"), PathBuf::from);
            let report = run_and_write(&cfg, &dir)?;
            println!("{}", report.text());
            println!("results_dir: {}", dir.display());
        }
        None => {
            let cfg = AbInitioConfig::default();
            let report = run_and_write(&cfg, "results-abinitio")?;
            println!("{}", report.text());
        }
    }
    Ok(())
}
