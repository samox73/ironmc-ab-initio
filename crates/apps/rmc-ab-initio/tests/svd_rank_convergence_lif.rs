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
fn second_order_energy_converges_with_pair_rank() {
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

    let svd = EphSvd::compress(&epmat, DEFAULT_TOL, MAX_RANK);
    let table =
        VertexTable::build(&epw, &crystal, &wigner, &svd, 8, DEFAULT_PHFREQ_CUTOFF_EV).unwrap();
    let ranks = [8, 12, 16, 25, 90, 100];
    let values = ranks.map(|rank| (e2(&table, true, &rank), e2(&table, false, &rank)));

    eprintln!("rank  E2_full[eV]  E2_short[eV]  delta_full[eV]  delta_short[eV]");
    for i in 0..ranks.len() {
        let previous = i.saturating_sub(1);
        eprintln!(
            "{:>4}  {:+.9}  {:+.9}  {:+.9}  {:+.9}",
            ranks[i],
            values[i].0,
            values[i].1,
            values[i].0 - values[previous].0,
            values[i].1 - values[previous].1,
        );
    }

    let plateau = (values[4].0 - values[5].0).abs();
    assert!(
        plateau < 0.002_f64.max(1e-3 * values[5].0.abs()),
        "rank 90→100 E² drift is {plateau:.6} eV"
    );
    let rank12_drift = (values[5].0 - values[1].0).abs() / values[5].0.abs();
    assert!(
        rank12_drift < 0.05,
        "rank 12→100 relative E² drift is {rank12_drift:.3}"
    );
}
