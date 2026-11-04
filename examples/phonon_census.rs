//! Census of interpolated phonon frequencies on the DMC q-grid.
//!
//! Diagnostic for the 1 meV soft-mode cutoff: on a healthy LiF 8³ grid
//! (Γ excluded) every branch should sit well above 1 meV, so the cutoff
//! should remove nothing. This prints how many (q, ν) actually fall below
//! the cutoff and the worst offenders, using .fmt data only.

use nalgebra::Vector3;
use rmc_ab_initio::epw::{Crystal, EpwData, Wigner};
use rmc_ab_initio::reconstruct::{RY_TO_EV, RY_TO_THZ, interpolate_d_polar};
use std::path::PathBuf;

const THZ_TO_MEV: f64 = RY_TO_EV / RY_TO_THZ * 1e3;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/LiF")
        .join(rel)
}

fn main() {
    let epw = EpwData::read(lif("epwdata.fmt")).unwrap();
    let crystal = Crystal::read(lif("crystal.fmt")).unwrap();
    let wigner = Wigner::read(lif("wigner.fmt")).unwrap();

    for n in [8usize, 12, 16] {
        let mut below_1mev = Vec::new();
        let mut min_acoustic = f64::INFINITY;
        let mut min_q = Vector3::zeros();
        let mut negative = 0usize;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if i == 0 && j == 0 && k == 0 {
                        continue;
                    }
                    let q = Vector3::new(
                        i as f64 / n as f64,
                        j as f64 / n as f64,
                        k as f64 / n as f64,
                    );
                    let ph = interpolate_d_polar(&epw, &crystal, &wigner, q);
                    for (nu, &w_thz) in ph.omega_thz.iter().enumerate() {
                        let w_mev = w_thz * THZ_TO_MEV;
                        if w_mev < 0.0 {
                            negative += 1;
                        }
                        if w_mev.abs() < 1.0 {
                            below_1mev.push((q, nu, w_mev));
                        }
                        if w_mev.abs() < min_acoustic {
                            min_acoustic = w_mev.abs();
                            min_q = q;
                        }
                    }
                }
            }
        }
        println!(
            "grid {n}^3: {} modes below 1 meV, {} negative omega, min |omega| = {:.4} meV at q = ({:.4},{:.4},{:.4})",
            below_1mev.len(),
            negative,
            min_acoustic,
            min_q[0],
            min_q[1],
            min_q[2]
        );
        for (q, nu, w) in below_1mev.iter().take(12) {
            println!(
                "   q=({:.4},{:.4},{:.4}) nu={nu} omega={w:.6} meV",
                q[0], q[1], q[2]
            );
        }
    }
}
