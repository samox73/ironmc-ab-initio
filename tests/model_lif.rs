//! A1 item 8: build the reduced artifact from the real LiF tensors and sanity-
//! check the physics — LO–TO splitting, Fröhlich 1/q² growth toward Γ, and a
//! serde round-trip. Cross-checks ω_LO(Γ) (closed form) against the epb `dynq`
//! (full DFPT dynamical matrix) when the coarse file is present.
//!
//! Needs the 146 MB `epmatwp`; skips when absent.

use nalgebra::Vector3;
use rmc_ab_initio::epb::Epb;
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::model::{regular_grid, ReducedModel};
use rmc_ab_initio::reconstruct::{hermitian_eig, interpolate_d_polar, GammaModel};
use std::path::PathBuf;

const RY_TO_EV: f64 = 13.605698066;
const RY_TO_THZ: f64 = 3289.84498;

/// Cartesian (2π/alat) → fractional crystal coords: q_frac[l] = a_l · q_cart.
fn to_frac(cry: &Crystal, qc: [f64; 3]) -> Vector3<f64> {
    Vector3::from_fn(|l, _| (0..3).map(|i| cry.at[l][i] * qc[i]).sum())
}

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/LiF")
        .join(rel)
}

fn load() -> Option<(EpwData, Crystal, Wigner, GammaModel)> {
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
    let g = GammaModel::build(&e, &c, &w, &wp);
    Some((e, c, w, g))
}

#[test]
fn reduced_model_physics_and_serde() {
    let Some((e, c, w, gamma)) = load() else {
        return;
    };

    // Fröhlich 1/q²: along a fixed direction, |g|² grows sharply toward Γ.
    let q = vec![
        Vector3::new(0.02, 0.0, 0.0),
        Vector3::new(0.1, 0.0, 0.0),
        Vector3::new(0.25, 0.25, 0.25),
    ];
    let m = ReducedModel::build(&gamma, &e, &c, &w, &q);
    assert!(
        m.g2_lo.iter().all(|&g| g.is_finite() && g > 0.0),
        "g² = {:?}",
        m.g2_lo
    );
    assert!(
        m.g2_lo[0] > m.g2_lo[1],
        "|g|² should grow toward Γ: {:?}",
        m.g2_lo
    );
    // 5× smaller q ⇒ ~25× larger |g|² (1/q²); allow slack for the finite short range.
    assert!(
        m.g2_lo[0] / m.g2_lo[1] > 10.0,
        "weak 1/q² growth: {}",
        m.g2_lo[0] / m.g2_lo[1]
    );

    // LO–TO splitting is real and positive; frequencies land in a sane range.
    // Physical ω_TO(Γ) ≈ 8.12 THz (33.6 meV, matches ph.x), ω_LO(Γ) ≈ 18.67 THz.
    assert!(
        m.omega_lo_mev > m.omega_to_mev,
        "no LO–TO splitting: {} vs {}",
        m.omega_lo_mev,
        m.omega_to_mev
    );
    assert!(
        (25.0..45.0).contains(&m.omega_to_mev),
        "ω_TO = {} meV",
        m.omega_to_mev
    );
    assert!(
        (60.0..95.0).contains(&m.omega_lo_mev),
        "ω_LO = {} meV",
        m.omega_lo_mev
    );
    assert!((m.cbm_ev - 9.6143).abs() < 2e-3);
    assert!((m.eps_inf - 2.0437).abs() < 1e-3);

    // serde cache round-trip.
    let tmp = std::env::temp_dir().join("rmc_reduced_lif.json");
    let full = ReducedModel::build(&gamma, &e, &c, &w, &regular_grid(6));
    full.save(&tmp).unwrap();
    assert_eq!(full, ReducedModel::load(&tmp).unwrap());
    assert_eq!(full.q.len(), 6 * 6 * 6 - 1);
    assert!(full.g2_lo.iter().all(|&g| g.is_finite() && g > 0.0));
    std::fs::remove_file(&tmp).ok();
    eprintln!(
        "A1.8: ω_TO={:.1} meV, ω_LO={:.1} meV, {} q-points",
        m.omega_to_mev,
        m.omega_lo_mev,
        full.q.len()
    );

    // ---- validate the polar phonons (interpolate_d_polar = short-range D +
    //      dyn_long_cart) against the coarse DFPT dynq, at the SAME q. Wannier
    //      interpolation is exact on the coarse grid and dyn_long_cart is the
    //      exact term EPW subtracted, so all six frequencies must match. This is
    //      the phonon analog of the A2 e-ph round-trip. ----
    let epb_path = lif("out/LiF.epb1");
    if !epb_path.exists() {
        return;
    }
    let epb = Epb::read(&epb_path, 20, 15, 6).unwrap();
    let iq = (0..epb.nqc)
        .filter(|&iq| epb.xqc[iq].iter().map(|x| x * x).sum::<f64>() > 1e-9)
        .min_by(|&a, &b| {
            let na: f64 = epb.xqc[a].iter().map(|x| x * x).sum();
            let nb: f64 = epb.xqc[b].iter().map(|x| x * x).sum();
            na.partial_cmp(&nb).unwrap()
        })
        .unwrap();
    let qf = to_frac(&c, epb.xqc[iq]);
    // epb dynq (full DFPT force constants) → frequencies.
    let mut d = epb.dynq_at(iq);
    for a in 0..6 {
        for b in 0..6 {
            d[a * 6 + b] /= (c.mass(a / 3) * c.mass(b / 3)).sqrt();
        }
    }
    let (w2, _) = hermitian_eig(&d, 6);
    let mut f_epb: Vec<f64> = w2.iter().map(|x| x.signum() * x.abs().sqrt()).collect();
    let mut f_ours = interpolate_d_polar(&e, &c, &w, qf)
        .omega_thz
        .iter()
        .map(|x| x / RY_TO_THZ)
        .collect::<Vec<_>>();
    f_epb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    f_ours.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo_mev = f_ours[5] * RY_TO_EV * 1000.0;
    eprintln!(
        "A1.8: polar phonons @ coarse |q| — LO ours={lo_mev:.1} meV vs epb LO={:.1} meV",
        f_epb[5] * RY_TO_EV * 1000.0
    );
    // compare the optical/LO branches (the ones the polar term shapes) tightly.
    for b in 3..6 {
        let rel = (f_ours[b] - f_epb[b]).abs() / f_epb[b];
        assert!(
            rel < 0.01,
            "polar phonon branch {b} disagrees with DFPT dynq: {:.2}%",
            rel * 100.0
        );
    }
}
