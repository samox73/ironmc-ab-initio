use rand::SeedableRng;
use rmc_ab_initio::eph_svd::{EphSvd, DEFAULT_TOL};
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::vertex_direct::DirectVertex;
use rmc_ab_initio::vertex_table::{
    e2, TablePrecision, VertexTable, DEFAULT_MAX_BYTES, DEFAULT_PHFREQ_CUTOFF_EV,
};
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
    let f64_table = VertexTable::build(
        &epw,
        &crystal,
        &wigner,
        &svd,
        8,
        DEFAULT_PHFREQ_CUTOFF_EV,
        TablePrecision::F64,
        DEFAULT_MAX_BYTES,
    )
    .unwrap();
    let f32_table = VertexTable::build(
        &epw,
        &crystal,
        &wigner,
        &svd,
        8,
        DEFAULT_PHFREQ_CUTOFF_EV,
        TablePrecision::F32,
        DEFAULT_MAX_BYTES,
    )
    .unwrap();
    let direct = DirectVertex::new(&epw, &crystal, &wigner, &epmat);
    let error = f64_table.max_direct_error(&direct, 200, 19);
    eprintln!("max compressed/direct vertex error={error:.3e}");
    assert!(error < 0.05, "max relative error {error:.3e}");

    let e2_delta = (e2(&f32_table, true, &usize::MAX) - e2(&f64_table, true, &usize::MAX)).abs();
    assert!(e2_delta < 1.0e-4, "f32 E2 drift is {} meV", e2_delta * 1e3);
    let mut rng = rand::rngs::StdRng::seed_from_u64(29);
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for _ in 0..1000 {
        let k = rand::Rng::gen_range(&mut rng, 0..f64_table.len());
        let q = rand::Rng::gen_range(&mut rng, 1..f64_table.len());
        let nu = rand::Rng::gen_range(&mut rng, 0..f64_table.nmodes);
        let expected = f64_table.full_indices(k, q, nu);
        worst = worst.max((f32_table.full_indices(k, q, nu) - expected).norm());
        scale = scale.max(expected.norm());
    }
    assert!(
        worst / scale < 1.0e-5,
        "max f32 vertex drift is {:.3e}",
        worst / scale
    );
}
