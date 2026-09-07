use std::path::PathBuf;

use indicatif::ProgressIterator;
use nalgebra::Vector3;
use rmc_ab_initio::{
    eph_svd::EphSvd,
    epw::{Crystal, EpmatWp, EpwData, Wigner},
    vertex_table::{e2, TablePrecision, VertexTable, DEFAULT_MAX_BYTES, DEFAULT_PHFREQ_CUTOFF_EV},
};
use rmc_core::time;
use tabled::{settings::Style, Table, Tabled};

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

#[derive(Tabled)]
struct Row {
    #[tabled(rename = "#")]
    max_rank: usize,
    e2_full: f64,
    e2_short: f64,
    delta_full: f64,
    delta_short: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let Some((e, c, w, wp)) = load() else {
        panic!("could not load data");
    };

    let grid_n = 8;
    let ranks: Vec<usize> = (1..300).step_by(10).collect();

    let svd = time!(EphSvd::compress; &wp, 1e-8, ranks[ranks.len() - 1]);
    log::info!(
        "max SVD reconstruction error: {:.3e}",
        svd.max_relative_error(&wp)
    );
    let table = time!(VertexTable::build; &e, &c, &w, &svd, grid_n, DEFAULT_PHFREQ_CUTOFF_EV, TablePrecision::F64, DEFAULT_MAX_BYTES)?;
    diagnose_second_order(&table, ranks.last().unwrap());

    let mut e2_fulls = Vec::with_capacity(ranks.len());
    let mut e2_shorts = Vec::with_capacity(ranks.len());
    for max_rank in ranks.iter().progress() {
        let full = e2(&table, true, max_rank);
        let short = e2(&table, false, max_rank);
        e2_fulls.push(full);
        e2_shorts.push(short);
    }

    let table = build_table(ranks, e2_fulls, e2_shorts);
    println!("{table}");

    Ok(())
}

fn diagnose_second_order(table: &VertexTable, rank: &usize) {
    let k0 = table.index_of(Vector3::zeros()).unwrap();
    let mut contributions = Vec::new();
    for q in 1..table.len() {
        let point = table.point(q);
        let neg_q = table.index_of(-point).unwrap();
        for nu in 0..table.nmodes {
            let g = table.full_indices_capped(k0, q, nu, rank);
            let denominator = -table.electron_ev[neg_q] - table.phonon_ev[q * table.nmodes + nu];
            contributions.push((g.norm_sqr() / denominator / table.len() as f64, point, nu));
        }
    }
    contributions.sort_by(|a, b| a.0.total_cmp(&b.0));
    eprintln!("20 largest E2 contributions:");
    for (value, q, nu) in contributions.iter().take(20) {
        eprintln!("q={q:?} nu={nu} contribution={value:.6} eV");
    }

    let step = 1.0 / table.grid_n as f64;
    let near = [
        Vector3::new(step, 0.0, 0.0),
        Vector3::new(-step, 0.0, 0.0),
        Vector3::new(0.0, step, 0.0),
        Vector3::new(0.0, -step, 0.0),
        Vector3::new(0.0, 0.0, step),
        Vector3::new(0.0, 0.0, -step),
    ]
    .into_iter()
    .map(|q| {
        let qi = table.index_of(q).unwrap();
        let nu = (0..table.nmodes)
            .max_by(|&a, &b| {
                table.long_ev[qi * table.nmodes + a]
                    .norm_sqr()
                    .total_cmp(&table.long_ev[qi * table.nmodes + b].norm_sqr())
            })
            .unwrap();
        contributions
            .iter()
            .find(|entry| table.index_of(entry.1) == Some(qi) && entry.2 == nu)
            .unwrap()
            .0
    })
    .sum::<f64>();
    eprintln!("six-nearest-point LO E2={near:.6} eV");
}

fn build_table(ranks: Vec<usize>, e2_fulls: Vec<f64>, e2_shorts: Vec<f64>) -> String {
    let rows = ranks
        .iter()
        .enumerate()
        .map(|(index, &max_rank)| Row {
            max_rank,
            e2_full: e2_fulls[index],
            e2_short: e2_shorts[index],
            delta_full: if index == 0 { 0. } else { e2_fulls[index] - e2_fulls[index - 1] },
            delta_short: if index == 0 { 0. } else { e2_shorts[index] - e2_shorts[index - 1] },
        })
        .collect::<Vec<_>>();

    let mut table = Table::new(rows);
    table.with(Style::rounded());
    table.to_string()
}
