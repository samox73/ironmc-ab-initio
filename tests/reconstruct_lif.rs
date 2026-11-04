//! Reconstruction parity against the real LiF dataset. `.fmt` files are
//! git-tracked (H/D checks always run); the g-reconstruction needs the 146 MB
//! `epmatwp` and skips when absent.

use nalgebra::Vector3;
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::reconstruct::{GammaModel, interpolate_d, interpolate_h};
use std::path::PathBuf;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/LiF")
        .join(rel)
}

#[test]
fn h_gamma_cbm_below_degenerate_p_manifold() {
    let e = EpwData::read(lif("epwdata.fmt")).unwrap();
    let w = Wigner::read(lif("wigner.fmt")).unwrap();
    let (ev, _u) = interpolate_h(&e, &w, Vector3::zeros());
    assert_eq!(ev.len(), 4);
    assert!((ev[0] - 9.6143).abs() < 1e-3, "CBM = {}", ev[0]); // Li-s, non-degenerate
    for k in 1..4 {
        assert!(
            (ev[k] - 24.3795).abs() < 1e-3,
            "p-manifold[{k}] = {}",
            ev[k]
        );
    }
    assert!(
        ev[1] - ev[0] > 10.0,
        "CBM should sit well below the p-manifold"
    );
}

#[test]
fn d_gamma_acoustic_sum_rule_and_to() {
    let e = EpwData::read(lif("epwdata.fmt")).unwrap();
    let c = Crystal::read(lif("crystal.fmt")).unwrap();
    let w = Wigner::read(lif("wigner.fmt")).unwrap();
    let mut om = interpolate_d(&e, &c, &w, Vector3::zeros()).omega_thz;
    om.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for k in 0..3 {
        assert!(om[k].abs() < 1e-2, "acoustic mode {k} not zero: {}", om[k]); // ASR
    }
    for k in 3..6 {
        // 3-fold TO; LO-TO splitting is subtracted here, restored by item 7.
        assert!(
            (om[k] - 11.9537).abs() < 1e-2,
            "optical mode {k}: {}",
            om[k]
        );
    }
}

/// The absolute CBM differs from FEP-DMC's pseudopotential dataset; only binding
/// energies relative to the CBM are comparable (`VertexTable::electron_ev` is shifted).
#[test]
fn lowest_band_follows_wannier_l_gamma_x_path() {
    let path = lif("LiF_band.dat");
    if !path.exists() {
        eprintln!(
            "skipping: {} absent (regenerate via run_pipeline.nu)",
            path.display()
        );
        return;
    }
    let epw = EpwData::read(lif("epwdata.fmt")).unwrap();
    let wigner = Wigner::read(lif("wigner.fmt")).unwrap();
    let raw = std::fs::read_to_string(path).unwrap();
    let reference: Vec<f64> = raw
        .lines()
        .take_while(|line| !line.trim().is_empty())
        .map(|line| line.split_whitespace().nth(1).unwrap().parse().unwrap())
        .collect();
    let kpoints: Vec<Vector3<f64>> = std::fs::read_to_string(lif("LiF_band.kpt"))
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| {
            let value: Vec<f64> = line
                .split_whitespace()
                .take(3)
                .map(|field| field.parse().unwrap())
                .collect();
            Vector3::new(value[0], value[1], value[2])
        })
        .collect();
    assert_eq!(reference.len(), kpoints.len());
    let max_error = (0..40)
        .map(|sample| sample * (kpoints.len() - 1) / 39)
        .map(|index| (interpolate_h(&epw, &wigner, kpoints[index]).0[0] - reference[index]).abs())
        .fold(0.0, f64::max);
    assert!(max_error < 5.0e-3, "lowest-band max error={max_error} eV");
}

#[test]
fn reduced_g_lo_finite_at_small_q() {
    let path = lif("out/LiF.epmatwp");
    if !path.exists() {
        eprintln!(
            "skipping: {} absent (regenerate via run_pipeline.nu)",
            path.display()
        );
        return;
    }
    let e = EpwData::read(lif("epwdata.fmt")).unwrap();
    let c = Crystal::read(lif("crystal.fmt")).unwrap();
    let w = Wigner::read(lif("wigner.fmt")).unwrap();
    let g = EpmatWp::read(&path, e.nbnd, e.nrr_k, e.nmodes, e.nrr_g).unwrap();
    if !g.data[0].is_finite() {
        // Guard against a regressed EPW run (ultrasoft pseudos → all-NaN e-ph; see
        // EPH-NAN-DIAGNOSIS.md). A norm-conserving run is finite and validates below.
        eprintln!(
            "skipping: {} is all-NaN; regenerate the EPW e-ph tensor",
            path.display()
        );
        return;
    }

    let model = GammaModel::build(&e, &c, &w, &g);
    assert!(
        (model.cbm_energy_ev - 9.6143).abs() < 1e-3,
        "CBM = {}",
        model.cbm_energy_ev
    );

    let (omega, g2) = model.reduced(&e, &c, &w, Vector3::new(0.05, 0.0, 0.0));
    let lo = (0..omega.len())
        .max_by(|&a, &b| omega[a].partial_cmp(&omega[b]).unwrap())
        .unwrap();
    assert!(omega[lo] > 10.0, "omega_LO = {}", omega[lo]);
    assert!(g2[lo].is_finite() && g2[lo] > 0.0, "|g_LO|^2 = {}", g2[lo]);
}
