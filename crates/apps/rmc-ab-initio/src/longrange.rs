//! The polar (Fröhlich) long-range e-ph vertex `g_L` — the one piece patt126
//! never had (its `fourierANDlr` is just a low-rank Fourier, no dipole term).
//!
//! EPW writes `epmatwp` with `g_L` **subtracted** (`lpolar=.true.`), so the
//! Wannier reconstruction in [`crate::reconstruct`] recovers only the short-range
//! part. `g = g_short + g_L`. This restores LiF's dominant 1/q coupling.
//!
//! A faithful port of EPW's `longrange.f90::rgd_blk_epw` (3D dipole branch),
//! Eq. (S4) of Verdi–Giustino PRL 115, 176401 (2015):
//!
//! ```text
//! g_L,κα(q) = i (4π e²/Ω) Σ_{G≠-q} e^{-(q+G)·ε∞·(q+G)/(4α·|2π/a|²)}
//!             · e^{-i(q+G)·τ_κ} · (q+G)·Z*_κ,·α / [(q+G)·ε∞·(q+G)]
//! ```
//!
//! Returned in the **Cartesian atomic-displacement basis** (`imode = 3κ+α`), in
//! **Ry**, matching how EPW subtracts it — i.e. directly additive to the
//! short-range Cartesian coupling from [`crate::reconstruct::GammaModel::g_cart`],
//! *before* any phonon-branch projection or `1/√(2Mω)` mode factor. Validated
//! against the coarse `.epb` (full coupling) in `tests/epb_lif.rs`.

use crate::epw::{Crystal, EpwData};
use num_complex::Complex64;

const E2: f64 = 2.0; // e² in Rydberg atomic units
const ALPH: f64 = 1.0; // Ewald convergence parameter (EPW default)
const GMAX: f64 = 14.0; // neglect e^{-x} terms with x > gmax (EPW default)

/// Long-range polar vertex `g_L,κα(q)` per Cartesian atomic-displacement mode
/// (`κα` = `3κ+α`), in Ry. `q` is fractional (crystal) coordinates, matching the
/// rest of the crate; it is folded into the reciprocal-space G-sum.
pub fn g_long_cart(cry: &Crystal, epw: &EpwData, q: [f64; 3]) -> Vec<Complex64> {
    let tpba = std::f64::consts::TAU / cry.alat; // 2π/a  (bohr⁻¹)
    let metric = tpba * tpba;
    let fac = E2 * 4.0 * std::f64::consts::PI / cry.omega; // 4π e²/Ω
    let eps = &epw.epsi;

    // per-direction G-box; the Ewald cutoff converges the sum well within it.
    // ponytail: symmetric box around 0, no min-|q+G| centering — exact enough
    //   for coarse q in the first BZ (validated to cos>0.99 vs epb). Add the
    //   EPW recentring only if we ever need g_L deep in higher BZs.
    let nrx: Vec<i64> = (0..3)
        .map(|j| {
            let bn = (cry.bg[j][0].powi(2) + cry.bg[j][1].powi(2) + cry.bg[j][2].powi(2)).sqrt();
            ((GMAX * ALPH * 4.0).sqrt() / bn).floor() as i64 + 2
        })
        .collect();

    let mut g = vec![Complex64::new(0.0, 0.0); cry.nmodes];
    for m1 in -nrx[0]..=nrx[0] {
        for m2 in -nrx[1]..=nrx[1] {
            for m3 in -nrx[2]..=nrx[2] {
                // (q+G) cartesian, bohr⁻¹: gg = (2π/a) Σ_j (m_j + q_j) bg_j
                let mut gg = [0.0; 3];
                for i in 0..3 {
                    gg[i] = tpba
                        * ((m1 as f64 + q[0]) * cry.bg[0][i]
                            + (m2 as f64 + q[1]) * cry.bg[1][i]
                            + (m3 as f64 + q[2]) * cry.bg[2][i]);
                }
                // qeq = (q+G)·ε∞·(q+G)
                let qeq = (0..3)
                    .map(|a| gg[a] * (0..3).map(|b| eps[a][b] * gg[b]).sum::<f64>())
                    .sum::<f64>();
                let ewald = qeq / (metric * ALPH * 4.0);
                if qeq <= 0.0 || ewald >= GMAX {
                    continue; // skip q+G=0 and the negligible Ewald tail
                }
                let facqd = fac * (-ewald).exp() / qeq;

                for na in 0..cry.nat {
                    // e^{-i(q+G)·τ_κ}: EPW arg = -a·(gg·τ), τ in alat units.
                    let arg = -cry.alat * (0..3).map(|i| gg[i] * cry.tau[na][i]).sum::<f64>();
                    let facq = facqd * Complex64::from_polar(1.0, arg);
                    for i in 0..3 {
                        // zaq = (q+G)·Z*_κ  (Cartesian component i of the displacement)
                        let zaq: f64 = (0..3).map(|p| gg[p] * epw.zstar[na][p][i]).sum();
                        g[3 * na + i] += facq * Complex64::new(0.0, zaq); // i·zaq
                    }
                }
            }
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epw::Crystal;

    /// Build a minimal cubic LiF-like `Crystal`/`EpwData` for the divergence
    /// self-check (no dataset files needed): 2 atoms, ±1 Born charges, ε∞=2.
    fn toy() -> (Crystal, EpwData) {
        let cry = Crystal {
            nat: 2,
            nmodes: 6,
            at: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            bg: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            omega: 100.0,
            alat: 5.0,
            tau: vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            ityp: vec![1, 2],
            amass: vec![1.0, 1.0],
        };
        let mut zstar = vec![[[0.0; 3]; 3]; 2];
        for a in 0..3 {
            zstar[0][a][a] = 1.0;
            zstar[1][a][a] = -1.0;
        }
        let epw = EpwData {
            ef: 0.0,
            nbnd: 1,
            nrr_k: 0,
            nmodes: 6,
            nrr_q: 0,
            nrr_g: 0,
            zstar,
            epsi: [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            h: vec![],
            d: vec![],
        };
        (cry, epw)
    }

    fn norm(g: &[Complex64]) -> f64 {
        g.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
    }

    #[test]
    fn frohlich_diverges_toward_gamma() {
        let (cry, epw) = toy();
        // 1/q Fröhlich signature: ‖g_L‖ grows as q→0.
        let near = norm(&g_long_cart(&cry, &epw, [0.02, 0.0, 0.0]));
        let far = norm(&g_long_cart(&cry, &epw, [0.10, 0.0, 0.0]));
        assert!(near > far, "‖g_L‖ should grow toward Γ: near={near} far={far}");
        // roughly 1/q: halving q from 0.10→0.05 should ~double the norm.
        let mid = norm(&g_long_cart(&cry, &epw, [0.05, 0.0, 0.0]));
        let ratio = mid / far;
        assert!(ratio > 1.6 && ratio < 2.4, "expected ~1/q scaling, got ratio {ratio}");
    }

    #[test]
    fn charge_neutral_pairs_cancel_at_zone_center_limit() {
        // Opposite Born charges ⇒ the two atoms' dipoles partially cancel; the
        // net stays finite and nonzero (the polar coupling exists).
        let (cry, epw) = toy();
        let g = g_long_cart(&cry, &epw, [0.03, 0.01, 0.0]);
        assert!(g.iter().all(|c| c.re.is_finite() && c.im.is_finite()));
        assert!(norm(&g) > 0.0);
    }
}
