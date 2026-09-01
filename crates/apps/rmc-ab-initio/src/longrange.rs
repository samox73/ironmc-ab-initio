//! The polar (Fröhlich) long-range terms EPW subtracts with `lpolar=.true.` and
//! that we must add back — the pieces patt126 never had:
//!
//! * [`g_long_cart`] — the e-ph vertex `g_L`, EPW `longrange.f90::rgd_blk_epw`
//!   (charge-dipole). Restores LiF's dominant 1/q coupling.
//! * [`dyn_long_cart`] — the dipole-dipole dynamical-matrix term, EPW
//!   `longrange.f90::rgd_blk` (Gonze form). Restores the LO–TO splitting and the
//!   physical TO frequency (the Wannier-interpolated D has it subtracted too).
//!
//! Both are the 3D dipole branch (no quadrupole, no 2D) evaluated by the same
//! Ewald G-sum, in Cartesian atomic-displacement units, Ry. Validated against the
//! coarse `.epb` (full coupling) and `dynq` (full dynamical matrix) in the tests.

use crate::epw::{Crystal, EpwData, Wigner};
use crate::reconstruct::{interpolate_d_polar, Phonons, RY_TO_EV, RY_TO_THZ};
use nalgebra::Vector3;
use num_complex::Complex64;

const E2: f64 = 2.0; // e² in Rydberg atomic units
const ALPH: f64 = 1.0; // Ewald convergence parameter (EPW default)
const GMAX: f64 = 14.0; // neglect e^{-x} terms with x > gmax (EPW default)

type Cf = Complex64;

/// Per-direction G-box bound; the Ewald cutoff converges the sum within it.
/// ponytail: symmetric box around 0, no min-|q+G| centering — exact enough for
///   coarse q in the first BZ (validated vs epb). Add EPW's recentring only if we
///   ever need these deep in higher BZs.
fn g_box(cry: &Crystal) -> [i64; 3] {
    std::array::from_fn(|j| {
        let bn = (cry.bg[j][0].powi(2) + cry.bg[j][1].powi(2) + cry.bg[j][2].powi(2)).sqrt();
        ((GMAX * ALPH * 4.0).sqrt() / bn).floor() as i64 + 2
    })
}

/// Cartesian `(q+G) = (2π/a) Σ_j (m_j + q_j) bg_j`, bohr⁻¹ (q fractional).
pub fn gg_cart(cry: &Crystal, m: [i64; 3], q: Vector3<f64>) -> Vector3<f64> {
    let tpba = std::f64::consts::TAU / cry.alat;
    Vector3::from_fn(|i, _| {
        tpba * ((m[0] as f64 + q[0]) * cry.bg[0][i]
            + (m[1] as f64 + q[1]) * cry.bg[1][i]
            + (m[2] as f64 + q[2]) * cry.bg[2][i])
    })
}

/// `g·ε·g`.
fn eps_quad(eps: &[[f64; 3]; 3], g: Vector3<f64>) -> f64 {
    (0..3)
        .map(|a| g[a] * (0..3).map(|b| eps[a][b] * g[b]).sum::<f64>())
        .sum()
}

/// `(g·Z*_κ)` per Cartesian component: `[Σ_p g_p Z*[p][i]]_i`.
fn zdotg(z: &[[f64; 3]; 3], g: Vector3<f64>) -> Vector3<f64> {
    Vector3::from_fn(|i, _| (0..3).map(|p| g[p] * z[p][i]).sum())
}

/// `metric = (2π/a)²`, the Ewald cutoff normalization.
fn metric(cry: &Crystal) -> f64 {
    let tpba = std::f64::consts::TAU / cry.alat;
    tpba * tpba
}

/// Long-range polar vertex `g_L,κα(q)` per Cartesian atomic-displacement mode
/// (`κα = 3κ+α`), in Ry. `q` is fractional. Port of `rgd_blk_epw` (3D dipole).
pub fn g_long_cart(cry: &Crystal, epw: &EpwData, q: Vector3<f64>) -> Vec<Cf> {
    let met = metric(cry);
    let fac = E2 * 4.0 * std::f64::consts::PI / cry.omega; // 4π e²/Ω
    let nrx = g_box(cry);
    let mut g = vec![Cf::new(0.0, 0.0); cry.nmodes];
    for m1 in -nrx[0]..=nrx[0] {
        for m2 in -nrx[1]..=nrx[1] {
            for m3 in -nrx[2]..=nrx[2] {
                let gg = gg_cart(cry, [m1, m2, m3], q);
                let qeq = eps_quad(&epw.epsi, gg);
                if qeq <= 0.0 || qeq / (met * ALPH * 4.0) >= GMAX {
                    continue; // skip q+G=0 and the negligible Ewald tail
                }
                let facqd = fac * (-qeq / (met * ALPH * 4.0)).exp() / qeq;
                for na in 0..cry.nat {
                    // e^{-i(q+G)·τ_κ}: EPW arg = -a·(gg·τ).
                    let arg = -cry.alat * (0..3).map(|i| gg[i] * cry.tau[na][i]).sum::<f64>();
                    let facq = facqd * Cf::from_polar(1.0, arg);
                    let zaq = zdotg(&epw.zstar[na], gg); // (q+G)·Z*_κ
                    for i in 0..3 {
                        g[3 * na + i] += facq * Cf::new(0.0, zaq[i]); // i·zaq
                    }
                }
            }
        }
    }
    g
}

/// Quantized polar vertex projected onto physical phonon branches, in eV.
pub fn g_long_branch(cry: &Crystal, epw: &EpwData, wigner: &Wigner, q: Vector3<f64>) -> Vec<Cf> {
    let phonons = interpolate_d_polar(epw, cry, wigner, q);
    project_cartesian_branches(cry, &phonons, &g_long_cart(cry, epw, q), 0.0)
}

pub(crate) fn quantized_eigenvectors(
    cry: &Crystal,
    phonons: &Phonons,
    phfreq_cutoff_ev: f64,
) -> Vec<Cf> {
    let mut out = vec![Cf::new(0.0, 0.0); cry.nmodes * cry.nmodes];
    for nu in 0..cry.nmodes {
        let omega_ry = phonons.omega_thz[nu].abs() / RY_TO_THZ;
        let factor = if omega_ry * RY_TO_EV > phfreq_cutoff_ev {
            1.0 / (2.0 * omega_ry).sqrt()
        } else {
            0.0
        };
        for a in 0..cry.nmodes {
            out[nu * cry.nmodes + a] = phonons.evec[nu][a] * factor / cry.mass(a / 3).sqrt();
        }
    }
    out
}

pub(crate) fn project_cartesian_branches(
    cry: &Crystal,
    phonons: &Phonons,
    cartesian: &[Cf],
    phfreq_cutoff_ev: f64,
) -> Vec<Cf> {
    let evec = quantized_eigenvectors(cry, phonons, phfreq_cutoff_ev);
    (0..cry.nmodes)
        .map(|nu| {
            (0..cry.nmodes)
                .map(|a| cartesian[a] * evec[nu * cry.nmodes + a])
                .sum::<Cf>()
                * RY_TO_EV
        })
        .collect()
}

/// Long-range dipole-dipole contribution to the dynamical matrix (Gonze form,
/// EPW `rgd_blk`, `signe=+1`), row-major `nmodes×nmodes`, **force-constant** units
/// (Ry/bohr² — add to the short-range D(q) *before* dividing by masses). Two
/// pieces: the q≠0 term over all atom-pair blocks and the q=0 self-term on the
/// diagonal blocks (which restores the acoustic sum rule).
pub fn dyn_long_cart(cry: &Crystal, epw: &EpwData, q: Vector3<f64>) -> Vec<Cf> {
    let n = cry.nmodes;
    let met = metric(cry);
    let fac = E2 * 4.0 * std::f64::consts::PI / cry.omega; // e² 4π/Ω, signe=+1
    let nrx = g_box(cry);
    let mut dm = vec![Cf::new(0.0, 0.0); n * n];

    for m1 in -nrx[0]..=nrx[0] {
        for m2 in -nrx[1]..=nrx[1] {
            for m3 in -nrx[2]..=nrx[2] {
                let m = [m1, m2, m3];

                // --- q=0 self-term: gg = G, diagonal blocks (na,na), minus ---
                let gs = gg_cart(cry, m, Vector3::zeros());
                let geg = eps_quad(&epw.epsi, gs);
                if geg > 0.0 && geg / (met * ALPH * 4.0) < GMAX {
                    let facgd = fac * (-geg / (met * ALPH * 4.0)).exp() / geg;
                    for na in 0..cry.nat {
                        let zag = zdotg(&epw.zstar[na], gs); // (G·Z*_na)
                        let mut fnat = [Cf::new(0.0, 0.0); 3];
                        for nb in 0..cry.nat {
                            let arg = cry.alat
                                * (0..3)
                                    .map(|k| gs[k] * (cry.tau[na][k] - cry.tau[nb][k]))
                                    .sum::<f64>();
                            let ph = Cf::from_polar(1.0, arg);
                            let zcg = zdotg(&epw.zstar[nb], gs);
                            for j in 0..3 {
                                fnat[j] += zcg[j] * ph;
                            }
                        }
                        for i in 0..3 {
                            for j in 0..3 {
                                dm[(3 * na + i) * n + (3 * na + j)] -= facgd * zag[i] * fnat[j];
                            }
                        }
                    }
                }

                // --- q≠0 term: gg = G+q, all blocks (na,nb), plus ---
                let gq = gg_cart(cry, m, q);
                let geg = eps_quad(&epw.epsi, gq);
                if geg > 0.0 && geg / (met * ALPH * 4.0) < GMAX {
                    let facgd = fac * (-geg / (met * ALPH * 4.0)).exp() / geg;
                    for nb in 0..cry.nat {
                        let zbg = zdotg(&epw.zstar[nb], gq); // (q+G)·Z*_nb
                        for na in 0..cry.nat {
                            let zag = zdotg(&epw.zstar[na], gq); // (q+G)·Z*_na
                            let arg = cry.alat
                                * (0..3)
                                    .map(|k| gq[k] * (cry.tau[na][k] - cry.tau[nb][k]))
                                    .sum::<f64>();
                            let facg = facgd * Cf::from_polar(1.0, arg);
                            for i in 0..3 {
                                for j in 0..3 {
                                    dm[(3 * na + i) * n + (3 * nb + j)] += facg * (zag[i] * zbg[j]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    dm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epw::Crystal;

    /// Minimal cubic LiF-like `Crystal`/`EpwData` for the self-checks (no files).
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

    fn norm(g: &[Cf]) -> f64 {
        g.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
    }

    #[test]
    fn frohlich_diverges_toward_gamma() {
        let (cry, epw) = toy();
        let near = norm(&g_long_cart(&cry, &epw, Vector3::new(0.02, 0.0, 0.0)));
        let far = norm(&g_long_cart(&cry, &epw, Vector3::new(0.10, 0.0, 0.0)));
        assert!(
            near > far,
            "‖g_L‖ should grow toward Γ: near={near} far={far}"
        );
        let mid = norm(&g_long_cart(&cry, &epw, Vector3::new(0.05, 0.0, 0.0)));
        let ratio = mid / far;
        assert!(
            ratio > 1.6 && ratio < 2.4,
            "expected ~1/q scaling, got ratio {ratio}"
        );
    }

    #[test]
    fn dyn_long_is_hermitian() {
        // The long-range dynamical matrix must be Hermitian (it is added to a
        // Hermitian force-constant matrix).
        let (cry, epw) = toy();
        let d = dyn_long_cart(&cry, &epw, Vector3::new(0.07, 0.03, 0.0));
        for i in 0..6 {
            for j in 0..6 {
                let asym = (d[i * 6 + j] - d[j * 6 + i].conj()).norm();
                assert!(asym < 1e-9, "non-Hermitian at ({i},{j}): {asym}");
            }
        }
    }

    #[test]
    fn quantization_drops_modes_below_the_cutoff() {
        let (cry, _) = toy();
        let mut phonons = Phonons {
            omega_thz: vec![0.5e-3 / RY_TO_EV * RY_TO_THZ; cry.nmodes],
            evec: (0..cry.nmodes)
                .map(|nu| {
                    (0..cry.nmodes)
                        .map(|a| Cf::new(f64::from(a == nu), 0.0))
                        .collect()
                })
                .collect(),
        };
        assert!(quantized_eigenvectors(&cry, &phonons, 1.0e-3)
            .iter()
            .all(|value| *value == Cf::new(0.0, 0.0)));
        phonons.omega_thz.fill(2.0e-3 / RY_TO_EV * RY_TO_THZ);
        assert!(quantized_eigenvectors(&cry, &phonons, 1.0e-3)
            .iter()
            .any(|value| value.norm() > 0.0));
    }
}
