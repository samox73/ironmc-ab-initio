//! Reader parity against the real LiF dataset in `datasets/LiF/`
//! (8×8×8 coarse-grid provenance since 2026-09-04, aligned with FEP-DMC).
//! `epwdata.fmt` and `crystal.fmt` are git-tracked; `out/LiF.epmatwp` (687 MB)
//! is regenerated, so that check skips when the file is absent.

use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData};
use std::path::PathBuf;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("datasets/LiF").join(rel)
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} != {b}");
}

#[test]
fn epwdata_header_dims_ef_polar() {
    let e = EpwData::read(lif("epwdata.fmt")).unwrap();

    // dims (Wigner–Seitz grids of the 8×8×8 coarse mesh)
    assert_eq!((e.nbnd, e.nrr_k, e.nmodes, e.nrr_q, e.nrr_g), (4, 617, 6, 833, 725));
    // Fermi energy (Ry)
    close(e.ef, 5.257_179_972_899_102_3e-2);

    // Born charges: Li +1.05, F -1.05 (diagonal, isotropic)
    assert_eq!(e.zstar.len(), 2);
    close(e.zstar[0][0][0], 1.054_549_906_134_500_1);
    close(e.zstar[1][0][0], -1.054_549_906_134_500_1);
    // ε∞ = 2.04
    close(e.epsi[0][0], 2.043_717_745_405);

    // complex streams sized to the dims
    assert_eq!(e.h.len(), e.nrr_k * e.nbnd * e.nbnd);
    assert_eq!(e.d.len(), e.nrr_q * e.nmodes * e.nmodes);
}

#[test]
fn crystal_geometry_masses_positions() {
    let c = Crystal::read(lif("crystal.fmt")).unwrap();

    assert_eq!((c.nat, c.nmodes), (2, 6));
    close(c.omega, 112.990_080_000_041_32); // bohr³
    close(c.alat, 7.674_205_698_939_999_8); // bohr

    // masses (Ry a.u.): Li then F
    close(c.mass(0), 6326.334_491_417_183_9);
    close(c.mass(1), 17315.617_730_578_255);

    // positions: Li at origin, F at (-1/2, 1/2, 1/2) alat
    close(c.tau[0][0], 0.0);
    close(c.tau[1][0], -0.5);
    close(c.tau[1][1], 0.5);
    close(c.tau[1][2], 0.5);
}

#[test]
fn epmatwp_loads_with_expected_extent() {
    let path = lif("out/LiF.epmatwp");
    if !path.exists() {
        eprintln!("skipping: {} not present (regenerate via run_pipeline.nu)", path.display());
        return;
    }
    let g = EpmatWp::read(&path, 4, 617, 6, 725).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(g.data.len(), EpmatWp::expected_len(4, 617, 6, 725));
    // last valid index must be addressable
    let _ = g.g(616, 724, 5, 3, 3);
}
