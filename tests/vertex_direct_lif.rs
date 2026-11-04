use nalgebra::Vector3;
use num_complex::Complex64;
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::longrange::g_long_cart;
use rmc_ab_initio::reconstruct::{GammaModel, interpolate_h};
use rmc_ab_initio::vertex_direct::DirectVertex;
use std::path::PathBuf;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/LiF")
        .join(rel)
}

fn max_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

#[test]
fn full_k_direct_matches_gamma_oracles() {
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

    let gamma = GammaModel::build(&epw, &crystal, &wigner, &epmat);
    let direct = DirectVertex::new(&epw, &crystal, &wigner, &epmat);
    assert!(
        max_diff(
            &direct.short_cart(Vector3::zeros(), Vector3::zeros()),
            &gamma.g_cart(Vector3::zeros())
        ) < 1e-8
    );

    let q = Vector3::new(0.125, 0.0, 0.0);
    let uq = &interpolate_h(&epw, &wigner, q).1[0];
    let short = gamma.g_cart_exact(q, uq);
    assert!(max_diff(&direct.short_cart(Vector3::zeros(), q), &short) < 1e-8);

    let overlap: Complex64 = uq.iter().zip(&gamma.u_cbm).map(|(a, b)| a.conj() * b).sum();
    let long = g_long_cart(&crystal, &epw, q);
    let expected: Vec<_> = short
        .iter()
        .zip(long)
        .map(|(s, l)| s + overlap * l)
        .collect();
    assert!(max_diff(&direct.full_cart(Vector3::zeros(), q), &expected) < 1e-8);
}
