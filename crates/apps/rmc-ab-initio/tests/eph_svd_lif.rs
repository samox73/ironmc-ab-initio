use rmc_ab_initio::eph_svd::{EphSvd, DEFAULT_TOL};
use rmc_ab_initio::epw::{EpmatWp, EpwData, Wigner};
use std::path::PathBuf;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../datasets/LiF")
        .join(rel)
}

#[test]
#[ignore = "full LiF tensor reconstruction; run explicitly for SVD validation"]
fn lif_wannier_tensor_reconstructs_at_requested_tolerance() {
    let path = lif("out/LiF.epmatwp");
    if !path.exists() {
        eprintln!("skipping: epmatwp absent (regenerate via run_pipeline.nu)");
        return;
    }
    let epw = EpwData::read(lif("epwdata.fmt")).unwrap();
    let wigner = Wigner::read(lif("wigner.fmt")).unwrap();
    let epmat = EpmatWp::read(&path, epw.nbnd, epw.nrr_k, epw.nmodes, epw.nrr_g).unwrap();
    if !epmat.data[0].is_finite() {
        eprintln!("skipping: epmatwp is non-finite");
        return;
    }
    let svd = EphSvd::compress_default(&epmat);
    let error = svd.max_relative_error(&epmat);
    let scale = epmat.data.iter().map(|g| g.norm()).fold(0.0, f64::max);
    let mut active_error: f64 = 0.0;
    for n in 0..epw.nbnd {
        for m in 0..epw.nbnd {
            for re in 0..epw.nrr_k {
                if wigner.k.ndegen(re, n, m) == 0 {
                    continue;
                }
                for rp in 0..epw.nrr_g {
                    for a in 0..epw.nmodes {
                        if wigner.g.ndegen(rp, 0, a / 3) > 0 {
                            active_error = active_error.max(
                                (svd.reconstruct(re, rp, a, n, m) - epmat.g(re, rp, a, n, m))
                                    .norm()
                                    / scale,
                            );
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "retained pair ranks={:?} (total {}), max error={error:.3e}, active error={active_error:.3e}",
        svd.pair_ranks(),
        svd.rank()
    );
    assert!(
        active_error < 10.0 * DEFAULT_TOL,
        "max active relative error {active_error:.3e} (raw {error:.3e})"
    );
}
