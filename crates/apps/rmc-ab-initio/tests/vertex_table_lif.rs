use rmc_ab_initio::eph_svd::{EphSvd, DEFAULT_TOL};
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::vertex_direct::DirectVertex;
use rmc_ab_initio::vertex_table::{VertexTable, DEFAULT_PHFREQ_CUTOFF_EV};
use std::path::PathBuf;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../datasets/LiF")
        .join(rel)
}

#[test]
#[ignore = "full LiF SVD plus an 8³ table and 200 direct evaluations"]
fn compressed_lif_vertex_has_no_wiring_errors() {
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
    let svd = EphSvd::compress_global(&epmat, DEFAULT_TOL, 300);
    let table =
        VertexTable::build(&epw, &crystal, &wigner, &svd, 8, DEFAULT_PHFREQ_CUTOFF_EV).unwrap();
    let direct = DirectVertex::new(&epw, &crystal, &wigner, &epmat);
    let error = table.max_direct_error(&direct, 200, 19);
    eprintln!("max compressed/direct vertex error={error:.3e}");
    assert!(error < 0.05, "max relative error {error:.3e}");
}
