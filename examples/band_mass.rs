//! Bare CBM dispersion ε(k) − ε(Γ) along the dispersion-sweep legs, for the
//! m_p/m_b normalization of the polaron mass (FEP-DMC's "mp/me" divides by the
//! band effective mass, not the free-electron mass), plus the phonon branches
//! ω_ν(q = p) at the same points: E₀(Γ) + ω_ν(p) is a state in the p sector
//! (polaron at Γ + one phonon at p), so its lowest branch bounds the one-phonon
//! continuum from above.
//!
//! Legs match `ironmc dispersion points`: x = s·(1,0,1), l = s·(1,1,1) in
//! fractional coordinates, with s = j/n on the grids and s → 0 for the
//! curvature mass. Prints JSON:
//!
//!   cargo run --release -p rmc-ab-initio --example band_mass out> runs/dispersion/band.json

use nalgebra::Vector3;
use rmc_ab_initio::epw::{Crystal, EpwData, Wigner};
use rmc_ab_initio::reconstruct::{RY_TO_EV, RY_TO_THZ, interpolate_d_polar, interpolate_h};
use std::path::PathBuf;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/LiF")
        .join(rel)
}

fn main() {
    let epw = EpwData::read(lif("epwdata.fmt")).unwrap();
    let crystal = Crystal::read(lif("crystal.fmt")).unwrap();
    let wigner = Wigner::read(lif("wigner.fmt")).unwrap();
    let thz_to_ev = RY_TO_EV / RY_TO_THZ;
    let cbm = |k: Vector3<f64>| interpolate_h(&epw, &wigner, k).0[0];
    let e_gamma = cbm(Vector3::zeros());

    let legs = [
        ("x", Vector3::new(1.0, 0.0, 1.0)),
        ("l", Vector3::new(1.0, 1.0, 1.0)),
    ];
    let mut rows = Vec::new();
    for (leg, dir) in legs {
        // Curvature at Γ: ε(s) = ε(Γ) + c·s² + O(s⁴); two tiny s give c without the quartic.
        let (s1, s2) = (1.0e-3, 2.0e-3);
        let (d1, d2) = (cbm(dir * s1) - e_gamma, cbm(dir * s2) - e_gamma);
        let c2 =
            (d1 * s2.powi(4) - d2 * s1.powi(4)) / (s1 * s1 * s2.powi(4) - s2 * s2 * s1.powi(4));
        rows.push(format!(
            r#"{{"leg":"{leg}","kind":"curvature","c2_per_s2":{c2:.10e}}}"#
        ));
        for n in [20usize, 40, 80, 120] {
            for j in 1..=n / 4 {
                let s = j as f64 / n as f64;
                let de = cbm(dir * s) - e_gamma;
                let omega: Vec<String> = interpolate_d_polar(&epw, &crystal, &wigner, dir * s)
                    .omega_thz
                    .iter()
                    .map(|w| format!("{:.10e}", w * thz_to_ev))
                    .collect();
                rows.push(format!(
                    r#"{{"leg":"{leg}","kind":"grid","n":{n},"j":{j},"de":{de:.10e},"omega":[{}]}}"#,
                    omega.join(",")
                ));
            }
        }
    }
    println!("[\n{}\n]", rows.join(",\n"));
}
