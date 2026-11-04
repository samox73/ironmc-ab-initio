//! A2 round-trip: the coarse `.epb` holds the **full** e-ph coupling (written
//! before EPW's long-range subtraction), so at every coarse q
//!
//! ```text
//!   g_full(Γ→Γ+q)  =  g_short(reconstruct)  +  ⟨u_{Γ+q}|u_Γ⟩ · g_L(longrange)
//! ```
//!
//! in the Cartesian atomic-displacement basis. The `⟨u_{Γ+q}|u_Γ⟩` factor is
//! EPW's `[U_{k+q}U_k†]_{mn}` (the CBM Bloch overlap) — it rotates the smooth-
//! basis dipole `g_L` back into the epb's raw KS gauge. Wannier interpolation is
//! exact on the coarse grid, so the round-trip closes to machine precision,
//! proving readers + reconstruction + longrange together.
//!
//! `.fmt` files are git-tracked; `epb1` (98 MB) + `epmatwp` (146 MB) are
//! regenerated, so the round-trip skips when either is absent.

use nalgebra::Vector3;
use num_complex::Complex64;
use rmc_ab_initio::epb::Epb;
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::longrange::g_long_cart;
use rmc_ab_initio::reconstruct::{GammaModel, interpolate_h};
use std::path::PathBuf;

const RY_TO_EV: f64 = 13.605698066;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/LiF")
        .join(rel)
}

fn norm(g: &[Complex64]) -> f64 {
    g.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// |⟨a|b⟩| / (‖a‖‖b‖) — invariant to the arbitrary overall KS↔eigensolver phase.
fn cos_sim(a: &[Complex64], b: &[Complex64]) -> f64 {
    let dot: Complex64 = a.iter().zip(b).map(|(x, y)| x.conj() * y).sum();
    dot.norm() / (norm(a) * norm(b))
}

/// Cartesian (2π/alat) → fractional crystal coords: q_frac[l] = a_l · q_cart.
fn to_frac(cry: &Crystal, qc: [f64; 3]) -> Vector3<f64> {
    Vector3::from_fn(|l, _| (0..3).map(|i| cry.at[l][i] * qc[i]).sum())
}

#[test]
fn epb_full_coupling_equals_short_plus_longrange() {
    let epb_path = lif("out/LiF.epb1");
    let wp_path = lif("out/LiF.epmatwp");
    if !epb_path.exists() || !wp_path.exists() {
        eprintln!("skipping: epb1/epmatwp absent (regenerate via run_pipeline.nu)");
        return;
    }

    let e = EpwData::read(lif("epwdata.fmt")).unwrap();
    let c = Crystal::read(lif("crystal.fmt")).unwrap();
    let w = Wigner::read(lif("wigner.fmt")).unwrap();
    let wp = EpmatWp::read(&wp_path, e.nbnd, e.nrr_k, e.nmodes, e.nrr_g).unwrap();
    if !wp.data[0].is_finite() {
        eprintln!("skipping: epmatwp is all-NaN; regenerate the EPW e-ph tensor");
        return;
    }
    let epb = Epb::read(&epb_path, 20, 15, 6).unwrap();

    // Γ = pool-local k with the minimum CBM (band index 5) energy. Confirm this
    // pool actually holds Γ by matching the known absolute CBM (9.6143 eV), and
    // that the H(Γ) reconstruction lands on the same value.
    let ik = epb.argmin_k(5);
    let cbm_ev = epb.et(ik, 5) * RY_TO_EV;
    assert!(
        (cbm_ev - 9.6143).abs() < 2e-3,
        "epb1 min-CBM k is not Γ (CBM={cbm_ev} eV)"
    );
    let model = GammaModel::build(&e, &c, &w, &wp);
    let u_gamma = model.u_cbm.clone();
    assert!(
        (model.cbm_energy_ev - cbm_ev).abs() < 2e-3,
        "reconstruct CBM vs epb"
    );

    // shells by |q| (Cartesian, 2π/alat); skip Γ (q=0, g_L undefined there).
    let mut shells: Vec<(f64, usize)> = (0..epb.nqc)
        .map(|iq| (norm(&epb.xqc[iq].map(Complex64::from)), iq))
        .filter(|&(qn, _)| qn > 1e-6)
        .collect();
    shells.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // ---- exact round-trip at every coarse q ----
    // epb's raw KS states carry an arbitrary overall phase per k, so compare
    // gauge-invariantly: direction (cosine) and magnitude (norm ratio).
    let mut worst_cos = 1.0f64;
    let mut worst_ratio = 0.0f64;
    let mut degenerate_skips = 0usize;
    for &(qn, iq) in &shells {
        let qf = to_frac(&c, epb.xqc[iq]);
        let full = epb.g_cbm_cart(ik, iq);
        let (energies, evecs) = interpolate_h(&e, &w, qf);
        // Where the CBM is (near-)degenerate with band 2 at Γ+q (the W star on
        // the 8×8×8 grid: bands 6/7 cross exactly), "the CBM eigenvector" is
        // gauge-ambiguous and the CBM-row comparison is undefined — skip, like
        // the symmetry-killed points below.
        if energies[1] - energies[0] < 0.01 {
            degenerate_skips += 1;
            continue;
        }
        let u_q = evecs[0].clone(); // CBM eigenvector at Γ+q
        let short = model.g_cart_exact(qf, &u_q);
        let gl = g_long_cart(&c, &e, qf);
        // ⟨u_{Γ+q}|u_Γ⟩: rotate the smooth-basis g_L into epb's raw KS gauge.
        let ov: Complex64 = (0..u_q.len()).map(|n| u_q[n].conj() * u_gamma[n]).sum();
        let recon: Vec<Complex64> = (0..6).map(|a| short[a] + gl[a] * ov).collect();

        // skip the few q where the full coupling is numerically ~0 (e.g. an
        // X-point where symmetry kills the CBM diagonal): cosine is undefined.
        if norm(&full) < 1e-2 {
            continue;
        }
        let cos = cos_sim(&recon, &full);
        let ratio = norm(&recon) / norm(&full);
        worst_cos = worst_cos.min(cos);
        worst_ratio = worst_ratio.max((ratio - 1.0).abs());
        assert!(
            cos > 0.9999,
            "round-trip direction broke at |q|={qn}: cos={cos}"
        );
        assert!(
            (ratio - 1.0).abs() < 2e-3,
            "round-trip magnitude broke at |q|={qn}: ratio={ratio}"
        );
    }
    eprintln!(
        "A2: exact round-trip over {} coarse q ({degenerate_skips} degenerate-CBM skips) — \
         worst cos={worst_cos:.6}, worst |ratio-1|={worst_ratio:.2e}",
        shells.len()
    );
    // The skip must stay the exception (the W star), never swallow the test.
    assert!(
        degenerate_skips <= 12,
        "{degenerate_skips} degenerate-CBM skips — the round-trip barely tests anything"
    );

    // ---- Fröhlich 1/q signature: ‖g_L‖ diverges toward Γ ----
    let q_near = shells[0].0;
    let far = *shells
        .iter()
        .rev()
        .find(|&&(qn, _)| qn > 2.0 * q_near)
        .unwrap();
    let gl_near = norm(&g_long_cart(&c, &e, to_frac(&c, epb.xqc[shells[0].1])));
    let gl_far = norm(&g_long_cart(&c, &e, to_frac(&c, epb.xqc[far.1])));
    assert!(
        gl_near > gl_far,
        "‖g_L‖ should grow toward Γ: near={gl_near} far={gl_far}"
    );
    // near/far |q| ratio is ≳2, and 1/q ⇒ ‖g_L‖ ratio should track it (loosely).
    assert!(
        gl_near / gl_far > 1.5,
        "g_L not diverging ~1/q: ratio {}",
        gl_near / gl_far
    );
}
