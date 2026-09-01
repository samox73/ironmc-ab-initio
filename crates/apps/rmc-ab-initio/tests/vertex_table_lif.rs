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
#[ignore = "full LiF SVD plus 200 direct vertex evaluations"]
fn compressed_lif_vertex_matches_direct_on_grid() {
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
    let svd = EphSvd::compress_default(&epmat);
    let table =
        VertexTable::build(&epw, &crystal, &wigner, &svd, 2, DEFAULT_PHFREQ_CUTOFF_EV).unwrap();
    let direct = DirectVertex::new(&epw, &crystal, &wigner, &epmat);
    let error = table.max_direct_error(&direct, 200, 19);
    eprintln!("max compressed/direct vertex error={error:.3e}");
    assert!(error < 10.0 * DEFAULT_TOL, "max relative error {error:.3e}");
}
