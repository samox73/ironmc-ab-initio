//! E⁽²⁾(long-range) vs q-grid size, table-free: validates that the small 8³
//! second-order energy is pure grid discretization (the Fröhlich 1/q² weight
//! is under-sampled on coarse grids) by extrapolating toward the continuum
//! weak-coupling limit ≈ −α·ω_LO ≈ −0.37 eV. Uses only .fmt data — no SVD,
//! no vertex table — so grids up to 32³ run in seconds.

use std::path::PathBuf;

use nalgebra::Vector3;
use rayon::prelude::*;
use rmc_ab_initio::epw::{Crystal, EpwData, Wigner};
use rmc_ab_initio::longrange::g_long_branch;
use rmc_ab_initio::reconstruct::{interpolate_d_polar, interpolate_h, RY_TO_EV, RY_TO_THZ};
use rmc_ab_initio::vertex_table::DEFAULT_PHFREQ_CUTOFF_EV;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/LiF")
        .join(rel)
}

fn main() {
    let epw = EpwData::read(lif("epwdata.fmt")).unwrap();
    let cry = Crystal::read(lif("crystal.fmt")).unwrap();
    let wig = Wigner::read(lif("wigner.fmt")).unwrap();

    let cbm = interpolate_h(&epw, &wig, Vector3::zeros()).0[0];

    println!("grid    1/n      e2_long[eV]");
    for n in [8usize, 10, 12, 14, 16, 20, 24, 32] {
        let nq = n * n * n;
        let sum: f64 = (1..nq)
            .into_par_iter()
            .map(|index| {
                let i = index / (n * n);
                let j = index / n % n;
                let k = index % n;
                let q = Vector3::new(
                    i as f64 / n as f64,
                    j as f64 / n as f64,
                    k as f64 / n as f64,
                );
                let eps_neg_q = interpolate_h(&epw, &wig, -q).0[0] - cbm;
                let g = g_long_branch(&cry, &epw, &wig, q, DEFAULT_PHFREQ_CUTOFF_EV);
                let ph = interpolate_d_polar(&epw, &cry, &wig, q);
                (0..cry.nmodes)
                    .map(|nu| {
                        let w_ev = ph.omega_thz[nu].abs() / RY_TO_THZ * RY_TO_EV;
                        g[nu].norm_sqr() / (-eps_neg_q - w_ev)
                    })
                    .sum::<f64>()
            })
            .sum();
        println!("{n:>4}  {:.4}   {:+.6}", 1.0 / n as f64, sum / nq as f64);
    }
}
