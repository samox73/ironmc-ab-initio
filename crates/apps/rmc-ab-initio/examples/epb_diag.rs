//! Diagnostic: correlate the epb round-trip cosine with the band-6/7 gap at
//! Γ+q, to identify gauge-ambiguous near-degenerate q points on the 8³ grid.

use nalgebra::Vector3;
use num_complex::Complex64;
use rmc_ab_initio::epb::Epb;
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::longrange::g_long_cart;
use rmc_ab_initio::reconstruct::{interpolate_h, GammaModel};
use std::path::PathBuf;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../datasets/LiF")
        .join(rel)
}

fn norm(g: &[Complex64]) -> f64 {
    g.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

fn cos_sim(a: &[Complex64], b: &[Complex64]) -> f64 {
    let dot: Complex64 = a.iter().zip(b).map(|(x, y)| x.conj() * y).sum();
    dot.norm() / (norm(a) * norm(b))
}

fn to_frac(cry: &Crystal, qc: [f64; 3]) -> Vector3<f64> {
    Vector3::from_fn(|l, _| (0..3).map(|i| cry.at[l][i] * qc[i]).sum())
}

fn main() {
    let e = EpwData::read(lif("epwdata.fmt")).unwrap();
    let c = Crystal::read(lif("crystal.fmt")).unwrap();
    let w = Wigner::read(lif("wigner.fmt")).unwrap();
    let wp = EpmatWp::read(&lif("out/LiF.epmatwp"), e.nbnd, e.nrr_k, e.nmodes, e.nrr_g).unwrap();
    let epb = Epb::read(&lif("out/LiF.epb1"), 20, 15, 6).unwrap();

    let ik = epb.argmin_k(5);
    let model = GammaModel::build(&e, &c, &w, &wp);
    let u_gamma = model.u_cbm.clone();

    println!("|q|      cos       gap12[eV]  |g_full|");
    for iq in 0..epb.nqc {
        let qn = epb.xqc[iq].iter().map(|x| x * x).sum::<f64>().sqrt();
        if qn < 1e-6 {
            continue;
        }
        let qf = to_frac(&c, epb.xqc[iq]);
        let full = epb.g_cbm_cart(ik, iq);
        if norm(&full) < 1e-2 {
            continue;
        }
        let (energies, evecs) = interpolate_h(&e, &w, qf);
        let gap = energies[1] - energies[0];
        let u_q = evecs[0].clone();
        let short = model.g_cart_exact(qf, &u_q);
        let gl = g_long_cart(&c, &e, qf);
        let ov: Complex64 = (0..u_q.len()).map(|n| u_q[n].conj() * u_gamma[n]).sum();
        let recon: Vec<Complex64> = (0..6).map(|a| short[a] + gl[a] * ov).collect();
        let cos = cos_sim(&recon, &full);
        if cos < 0.9999 {
            println!("{qn:.6}  {cos:.6}  {gap:9.4}  {:.4}", norm(&full));
        }
    }
    println!("(only q with cos < 0.9999 listed)");
}
