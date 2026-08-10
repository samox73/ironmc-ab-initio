//! Wannier→Bloch interpolation at k=Γ, reduced to what the DiagMC vertex needs.
//!
//! Mirrors the reconstruction machinery of patt126's `Wannier2Bloch.py`
//! (double Fourier + eigenvector rotation) but adds the two corrections the plan
//! calls out for LiF: EPW's per-pair Wigner-Seitz degeneracy weights (patt126
//! drops them), and a per-atom-pair mass factor `√(Mκ Mκ')` (patt126 hard-codes
//! Silicon's scalar mass).
//!
//! Reduced approximation (single band near Γ): both electron legs are rotated
//! with `U(Γ)`, so the vertex is a function of q only — `plan-rmc-ab-initio.md`.

use crate::epw::{Crystal, EpmatWp, EpwData, Wigner};
use num_complex::Complex64;

const RY_TO_EV: f64 = 13.605698066;
const RY_TO_THZ: f64 = 3289.84498;

type Cf = Complex64;
const C0: Cf = Complex64::new(0.0, 0.0);

/// exp(i 2π q·R) for fractional q and integer R.
fn phase(q: [f64; 3], r: [i64; 3]) -> Cf {
    let d = q[0] * r[0] as f64 + q[1] * r[1] as f64 + q[2] * r[2] as f64;
    Cf::from_polar(1.0, std::f64::consts::TAU * d)
}

/// Eigen-decomposition of an `n×n` complex-Hermitian matrix `a` (row-major),
/// via the real `2n×2n` symmetric embedding `[[Re,-Im],[Im,Re]]`. Returns
/// eigenvalues ascending and orthonormal eigenvectors (`evecs[k][orb]`).
///
/// The embedding doubles every eigenvalue; we recover the true `n` vectors by
/// Gram-Schmidt over the sorted candidates, which stays correct through
/// degeneracies (LiF's Γ has a 3-fold p-manifold and a 3-fold TO).
pub fn hermitian_eig(a: &[Cf], n: usize) -> (Vec<f64>, Vec<Vec<Cf>>) {
    let mut m = nalgebra::DMatrix::<f64>::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            let aij = a[i * n + j];
            m[(i, j)] = aij.re;
            m[(i + n, j + n)] = aij.re;
            m[(i, j + n)] = -aij.im;
            m[(i + n, j)] = aij.im;
        }
    }
    let se = m.symmetric_eigen();

    // (eigenvalue, complex reduction) candidates, sorted ascending.
    let mut cand: Vec<(f64, Vec<Cf>)> = (0..2 * n)
        .map(|c| {
            let col = se.eigenvectors.column(c);
            let v = (0..n).map(|k| Cf::new(col[k], col[k + n])).collect();
            (se.eigenvalues[c], v)
        })
        .collect();
    cand.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

    let mut evals = Vec::with_capacity(n);
    let mut evecs: Vec<Vec<Cf>> = Vec::with_capacity(n);
    for (lambda, mut v) in cand {
        if evecs.len() == n {
            break;
        }
        for u in &evecs {
            let proj = zdot(u, &v); // <u|v>
            for k in 0..n {
                v[k] -= proj * u[k];
            }
        }
        let norm = zdot(&v, &v).re.sqrt();
        if norm > 1e-7 {
            for k in 0..n {
                v[k] /= norm;
            }
            evals.push(lambda);
            evecs.push(v);
        }
    }
    (evals, evecs)
}

/// Hermitian inner product ⟨u|v⟩ = Σ conj(u)·v.
fn zdot(u: &[Cf], v: &[Cf]) -> Cf {
    u.iter().zip(v).map(|(a, b)| a.conj() * b).sum()
}

/// Force a matrix Hermitian (kills the tiny anti-Hermitian noise the Fourier sum
/// leaves behind), in place, row-major `n×n`.
fn hermitize(a: &mut [Cf], n: usize) {
    for i in 0..n {
        for j in i..n {
            let h = 0.5 * (a[i * n + j] + a[j * n + i].conj());
            a[i * n + j] = h;
            a[j * n + i] = h.conj();
        }
    }
}

/// Interpolated electron eigen-energies (eV, ascending) and eigenvectors at
/// fractional k. At k=Γ the lowest is the LiF CBM (Li-s), non-degenerate.
pub fn interpolate_h(epw: &EpwData, wig: &Wigner, k: [f64; 3]) -> (Vec<f64>, Vec<Vec<Cf>>) {
    let n = epw.nbnd;
    let mut h = vec![C0; n * n];
    for ir in 0..wig.k.nr() {
        let ph = phase(k, wig.k.irvec[ir]);
        for i in 0..n {
            for j in 0..n {
                let g = wig.k.ndegen(ir, i, j);
                if g > 0 {
                    h[i * n + j] += ph * epw.h_at(ir, i, j) / g as f64;
                }
            }
        }
    }
    hermitize(&mut h, n);
    let (evals, evecs) = hermitian_eig(&h, n);
    (evals.iter().map(|e| e * RY_TO_EV).collect(), evecs)
}

/// Interpolated phonons at fractional q: frequencies (THz; negative = imaginary)
/// and polarization vectors `evec[branch][cartesian-mode]`.
pub struct Phonons {
    pub omega_thz: Vec<f64>,
    pub evec: Vec<Vec<Cf>>,
}

pub fn interpolate_d(epw: &EpwData, cry: &Crystal, wig: &Wigner, q: [f64; 3]) -> Phonons {
    let n = epw.nmodes;
    let mut d = vec![C0; n * n];
    for ir in 0..wig.q.nr() {
        let ph = phase(q, wig.q.irvec[ir]);
        for a in 0..n {
            for b in 0..n {
                let g = wig.q.ndegen(ir, a / 3, b / 3);
                if g > 0 {
                    d[a * n + b] += ph * epw.d_at(ir, a, b) / g as f64;
                }
            }
        }
    }
    // force constants → dynamical matrix: divide by √(Mκ Mκ').
    for a in 0..n {
        for b in 0..n {
            d[a * n + b] /= (cry.mass(a / 3) * cry.mass(b / 3)).sqrt();
        }
    }
    hermitize(&mut d, n);
    let (w2, evec) = hermitian_eig(&d, n);
    let omega_thz = w2.iter().map(|w| w.signum() * w.abs().sqrt() * RY_TO_THZ).collect();
    Phonons { omega_thz, evec }
}

/// The reduced Γ model: everything q-independent is baked in once (needs the
/// 146 MB `epmatwp`), leaving a cheap per-q evaluation.
pub struct GammaModel {
    pub nmodes: usize,
    pub cbm: usize,
    pub cbm_energy_ev: f64,
    /// R_p vectors of the el-ph grid (for the per-q phonon Fourier).
    pub irvec_g: Vec<[i64; 3]>,
    /// `gbar[rp][cart-mode]`: electron legs already collapsed onto the CBM and
    /// both WS degeneracies folded in. Short-range only (long-range = item 7).
    pub gbar: Vec<Vec<Cf>>,
}

impl GammaModel {
    pub fn build(epw: &EpwData, _cry: &Crystal, wig: &Wigner, epmat: &EpmatWp) -> Self {
        let nbnd = epw.nbnd;
        let nmodes = epw.nmodes;
        let (evals, evecs) = interpolate_h(epw, wig, [0.0; 3]);
        let cbm = 0; // ascending → lowest = CBM
        let u = &evecs[cbm]; // u[orb]

        // gbar[rp][a] = Σ_{re,n,m} conj(u_n) g(re,rp,a,n,m) u_m
        //              / ( ndegen_k(re,n,m) · ndegen_g(rp,0,a/3) )
        // ndegen_g is iw-independent for LiF (all WFs co-centered on Li), so we
        // index its electron slot at 0 after the CBM rotation.
        // ponytail: iw=0 slot is LiF-specific; a multi-centre material would
        //   apply ndegen_g before rotating the electron legs.
        let mut gbar = vec![vec![C0; nmodes]; wig.g.nr()];
        for rp in 0..wig.g.nr() {
            for a in 0..nmodes {
                let dg = wig.g.ndegen(rp, 0, a / 3);
                if dg <= 0 {
                    continue;
                }
                let mut acc = C0;
                for re in 0..epw.nrr_k {
                    for n in 0..nbnd {
                        for m in 0..nbnd {
                            let dk = wig.k.ndegen(re, n, m);
                            if dk > 0 {
                                acc += u[n].conj() * epmat.g(re, rp, a, n, m) * u[m] / dk as f64;
                            }
                        }
                    }
                }
                gbar[rp][a] = acc / dg as f64;
            }
        }
        GammaModel { nmodes, cbm, cbm_energy_ev: evals[cbm], irvec_g: wig.g.irvec.clone(), gbar }
    }

    /// Short-range coupling per phonon branch at fractional q (eV), CBM band.
    pub fn g_branches(&self, q: [f64; 3], ph: &Phonons) -> Vec<Cf> {
        // Cartesian-mode coupling: g_cart[a] = Σ_rp e^{i2π q·R_p} gbar[rp][a].
        let mut g_cart = vec![C0; self.nmodes];
        for rp in 0..self.irvec_g.len() {
            let f = phase(q, self.irvec_g[rp]);
            for a in 0..self.nmodes {
                g_cart[a] += f * self.gbar[rp][a];
            }
        }
        // project onto branches with the polarization vectors.
        (0..self.nmodes)
            .map(|v| {
                let g: Cf = (0..self.nmodes).map(|a| g_cart[a] * ph.evec[v][a]).sum();
                g * RY_TO_EV
            })
            .collect()
    }

    /// `(ω_ν THz, |g_ν|² eV²)` per branch at fractional q, short-range only.
    pub fn reduced(&self, epw: &EpwData, cry: &Crystal, wig: &Wigner, q: [f64; 3]) -> (Vec<f64>, Vec<f64>) {
        let ph = interpolate_d(epw, cry, wig, q);
        let g2 = self.g_branches(q, &ph).iter().map(|g| g.norm_sqr()).collect();
        (ph.omega_thz.clone(), g2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermitian_eig_reconstructs_with_degeneracy() {
        // 3×3 Hermitian with a deliberate 2-fold degenerate eigenvalue.
        let a = vec![
            Cf::new(2.0, 0.0), Cf::new(0.0, 1.0), Cf::new(0.0, 0.0),
            Cf::new(0.0, -1.0), Cf::new(2.0, 0.0), Cf::new(0.0, 0.0),
            Cf::new(0.0, 0.0), Cf::new(0.0, 0.0), Cf::new(5.0, 0.0),
        ];
        let (evals, evecs) = hermitian_eig(&a, 3);
        assert_eq!(evals.len(), 3);
        // reconstruct A = Σ λ_k v_k v_k† (basis-independent, survives degeneracy)
        let mut r = vec![C0; 9];
        for k in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    r[i * 3 + j] += evals[k] * evecs[k][i] * evecs[k][j].conj();
                }
            }
        }
        for idx in 0..9 {
            assert!((r[idx] - a[idx]).norm() < 1e-9, "entry {idx}: {} vs {}", r[idx], a[idx]);
        }
    }

    #[test]
    fn g_branches_fourier_sums_and_projects() {
        // 1 branch, 2 R_p vectors, trivial polarization: exercises the per-q
        // phase, Fourier sum, branch projection and Ry→eV scale on known inputs.
        // (The full gbar build needs real epmatwp, which A0 shipped as NaN.)
        let model = GammaModel {
            nmodes: 1,
            cbm: 0,
            cbm_energy_ev: 0.0,
            irvec_g: vec![[0, 0, 0], [1, 0, 0]],
            gbar: vec![vec![Cf::new(2.0, 0.0)], vec![Cf::new(0.0, 1.0)]],
        };
        let ph = Phonons { omega_thz: vec![5.0], evec: vec![vec![Cf::new(1.0, 0.0)]] };
        // q=1/4 ⇒ phase at R_p=(1,0,0) is e^{iπ/2}=i, so g_cart = 2 + i·i = 1.
        let g = model.g_branches([0.25, 0.0, 0.0], &ph);
        assert!((g[0] - Cf::new(RY_TO_EV, 0.0)).norm() < 1e-12, "{}", g[0]);
    }
}
