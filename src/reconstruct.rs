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
use nalgebra::Vector3;
use num_complex::Complex64;

pub const RY_TO_EV: f64 = 13.605698066;
pub const RY_TO_THZ: f64 = 3289.84498;

type Cf = Complex64;
const C0: Cf = Complex64::new(0.0, 0.0);

/// exp(i 2π q·R) for fractional q and integer R.
pub(crate) fn phase(q: Vector3<f64>, r: [i64; 3]) -> Cf {
    let d = q[0] * r[0] as f64 + q[1] * r[1] as f64 + q[2] * r[2] as f64;
    Cf::from_polar(1.0, std::f64::consts::TAU * d)
}

/// Eigen-decomposition of an `n×n` complex-Hermitian matrix `a` (row-major).
/// Returns eigenvalues ascending and orthonormal eigenvectors (`evecs[k][orb]`).
pub fn hermitian_eig(a: &[Cf], n: usize) -> (Vec<f64>, Vec<Vec<Cf>>) {
    assert_eq!(a.len(), n * n);
    let m = nalgebra::DMatrix::from_fn(n, n, |i, j| a[i * n + j]);
    let se = m.symmetric_eigen();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| se.eigenvalues[i].partial_cmp(&se.eigenvalues[j]).unwrap());
    let evals = order.iter().map(|&i| se.eigenvalues[i]).collect();
    let evecs = order
        .iter()
        .map(|&i| se.eigenvectors.column(i).iter().copied().collect())
        .collect();
    (evals, evecs)
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
pub fn interpolate_h(epw: &EpwData, wig: &Wigner, k: Vector3<f64>) -> (Vec<f64>, Vec<Vec<Cf>>) {
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

/// Short-range force-constant matrix at fractional q (Wannier Fourier of D(R)),
/// row-major `nmodes²`, Ry/bohr². EPW subtracted the polar long-range before
/// Wannierizing, so this is short-range only — see [`interpolate_d_polar`].
fn dyn_force_constants(epw: &EpwData, wig: &Wigner, q: Vector3<f64>) -> Vec<Cf> {
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
    d
}

/// Mass-reduce force constants (÷√(Mκ Mκ')), diagonalize → frequencies (THz;
/// negative = imaginary) and polarization vectors.
fn phonons_from_fc(mut d: Vec<Cf>, cry: &Crystal, n: usize) -> Phonons {
    for a in 0..n {
        for b in 0..n {
            d[a * n + b] /= (cry.mass(a / 3) * cry.mass(b / 3)).sqrt();
        }
    }
    hermitize(&mut d, n);
    let (w2, evec) = hermitian_eig(&d, n);
    let omega_thz = w2
        .iter()
        .map(|w| w.signum() * w.abs().sqrt() * RY_TO_THZ)
        .collect();
    Phonons { omega_thz, evec }
}

/// **Short-range** phonons: the Wannier-interpolated D(q) alone, with the polar
/// LO–TO splitting still subtracted (the optical triplet stays degenerate at Γ).
/// Use [`interpolate_d_polar`] for physical frequencies.
pub fn interpolate_d(epw: &EpwData, cry: &Crystal, wig: &Wigner, q: Vector3<f64>) -> Phonons {
    phonons_from_fc(dyn_force_constants(epw, wig, q), cry, epw.nmodes)
}

/// **Physical** phonons: short-range D(q) plus the polar dipole-dipole long-range
/// ([`crate::longrange::dyn_long_cart`]), restoring the true TO frequency and the
/// LO–TO splitting at finite q. Matches the coarse DFPT `dynq`.
pub fn interpolate_d_polar(epw: &EpwData, cry: &Crystal, wig: &Wigner, q: Vector3<f64>) -> Phonons {
    let mut d = dyn_force_constants(epw, wig, q);
    let dl = crate::longrange::dyn_long_cart(cry, epw, q);
    for (a, b) in d.iter_mut().zip(&dl) {
        *a += *b;
    }
    phonons_from_fc(d, cry, epw.nmodes)
}

/// The reduced Γ model: everything q-independent is baked in once (needs the
/// 146 MB `epmatwp`), leaving a cheap per-q evaluation.
pub struct GammaModel {
    pub nmodes: usize,
    pub cbm: usize,
    pub cbm_energy_ev: f64,
    /// R_p vectors of the el-ph grid (for the per-q phonon Fourier).
    pub irvec_g: Vec<[i64; 3]>,
    /// CBM eigenvector at Γ (the ket electron leg, `u[orb]`).
    pub u_cbm: Vec<Cf>,
    /// `gbar_open[rp][bra_orb][cart]`: the **ket** leg collapsed onto the Γ CBM
    /// and both WS degeneracies folded in, but the **bra** electron leg left
    /// open. Short-range only (long-range = [`crate::longrange`]). Collapsing
    /// the bra with `u_cbm` too gives the reduced Γ-vertex ([`Self::g_cart`]);
    /// collapsing it with the CBM at k+q gives the exact vertex
    /// ([`Self::g_cart_exact`]) — needed only to validate the round-trip.
    pub gbar_open: Vec<Vec<Vec<Cf>>>,
}

impl GammaModel {
    pub fn build(epw: &EpwData, _cry: &Crystal, wig: &Wigner, epmat: &EpmatWp) -> Self {
        let nbnd = epw.nbnd;
        let nmodes = epw.nmodes;
        let (evals, evecs) = interpolate_h(epw, wig, Vector3::zeros());
        let cbm = 0; // ascending → lowest = CBM
        let u = evecs[cbm].clone(); // u[orb]

        // gbar_open[rp][n][a] = Σ_{re,m} g(re,rp,a,n,m) u_m
        //                       / ( ndegen_k(re,n,m) · ndegen_g(rp,0,a/3) )
        // ket leg collapsed on the Γ CBM; bra leg n left open.
        // ndegen_g is iw-independent for LiF (all WFs co-centered on Li), so we
        // index its electron slot at 0 after the ket rotation.
        // ponytail: iw=0 slot is LiF-specific; a multi-centre material would
        //   apply ndegen_g before rotating the electron legs.
        let mut gbar_open = vec![vec![vec![C0; nmodes]; nbnd]; wig.g.nr()];
        for rp in 0..wig.g.nr() {
            for a in 0..nmodes {
                let dg = wig.g.ndegen(rp, 0, a / 3);
                if dg <= 0 {
                    continue;
                }
                for n in 0..nbnd {
                    let mut acc = C0;
                    for re in 0..epw.nrr_k {
                        for m in 0..nbnd {
                            let dk = wig.k.ndegen(re, n, m);
                            if dk > 0 {
                                acc += epmat.g(re, rp, a, n, m) * u[m] / dk as f64;
                            }
                        }
                    }
                    gbar_open[rp][n][a] = acc / dg as f64;
                }
            }
        }
        GammaModel {
            nmodes,
            cbm,
            cbm_energy_ev: evals[cbm],
            irvec_g: wig.g.irvec.clone(),
            u_cbm: u,
            gbar_open,
        }
    }

    /// Fourier the open-bra tensor to fractional q: `Σ_{R_p} e^{i2π q·R_p}
    /// gbar_open[R_p][n][a]` → `[bra_orb][cart]`.
    fn g_open_at(&self, q: Vector3<f64>) -> Vec<Vec<Cf>> {
        let mut out = vec![vec![C0; self.nmodes]; self.u_cbm.len()];
        for rp in 0..self.irvec_g.len() {
            let f = phase(q, self.irvec_g[rp]);
            for n in 0..self.u_cbm.len() {
                for a in 0..self.nmodes {
                    out[n][a] += f * self.gbar_open[rp][n][a];
                }
            }
        }
        out
    }

    /// Contract the open bra leg with `u_bra` → Cartesian coupling `[cart]` (Ry).
    fn contract_bra(&self, open: &[Vec<Cf>], u_bra: &[Cf]) -> Vec<Cf> {
        (0..self.nmodes)
            .map(|a| (0..u_bra.len()).map(|n| u_bra[n].conj() * open[n][a]).sum())
            .collect()
    }

    /// Reduced short-range coupling in the **Cartesian atomic-displacement
    /// basis** (`imode = 3κ+α`), CBM band, in **Ry** — the frame in which EPW
    /// subtracts the long-range part, so [`crate::longrange::g_long_cart`] adds
    /// to this directly. Both electron legs use the Γ CBM (the reduced Γ-model).
    pub fn g_cart(&self, q: Vector3<f64>) -> Vec<Cf> {
        let u = self.u_cbm.clone();
        self.contract_bra(&self.g_open_at(q), &u)
    }

    /// Exact short-range coupling `g(Γ→Γ+q)`: ket = Γ CBM, **bra = CBM at k+q**
    /// (`u_bra` from `interpolate_h(q)`). Resolves the electron-leg
    /// k-dependence the reduced [`Self::g_cart`] drops — used to close the A2
    /// round-trip against the coarse `.epb`. Cartesian basis, Ry.
    pub fn g_cart_exact(&self, q: Vector3<f64>, u_bra: &[Cf]) -> Vec<Cf> {
        self.contract_bra(&self.g_open_at(q), u_bra)
    }

    /// Short-range coupling per phonon branch at fractional q (eV), CBM band.
    pub fn g_branches(&self, q: Vector3<f64>, ph: &Phonons) -> Vec<Cf> {
        let g_cart = self.g_cart(q);
        // project onto branches with the polarization vectors.
        (0..self.nmodes)
            .map(|v| {
                let g: Cf = (0..self.nmodes).map(|a| g_cart[a] * ph.evec[v][a]).sum();
                g * RY_TO_EV
            })
            .collect()
    }

    /// `(ω_ν THz, |g_ν|² eV²)` per branch at fractional q, short-range only.
    pub fn reduced(
        &self,
        epw: &EpwData,
        cry: &Crystal,
        wig: &Wigner,
        q: Vector3<f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        let ph = interpolate_d(epw, cry, wig, q);
        let g2 = self
            .g_branches(q, &ph)
            .iter()
            .map(|g| g.norm_sqr())
            .collect();
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
            Cf::new(2.0, 0.0),
            Cf::new(0.0, 1.0),
            Cf::new(0.0, 0.0),
            Cf::new(0.0, -1.0),
            Cf::new(2.0, 0.0),
            Cf::new(0.0, 0.0),
            Cf::new(0.0, 0.0),
            Cf::new(0.0, 0.0),
            Cf::new(5.0, 0.0),
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
            assert!(
                (r[idx] - a[idx]).norm() < 1e-9,
                "entry {idx}: {} vs {}",
                r[idx],
                a[idx]
            );
        }
    }

    #[test]
    fn g_branches_fourier_sums_and_projects() {
        // 1 branch, 2 R_p vectors, trivial polarization: exercises the per-q
        // phase, Fourier sum, branch projection and Ry→eV scale on known inputs.
        // (The full gbar build needs real epmatwp, which A0 shipped as NaN.)
        // 1 band ⇒ u_cbm=[1], so gbar_open[rp][0][0] is the Cartesian coupling.
        let model = GammaModel {
            nmodes: 1,
            cbm: 0,
            cbm_energy_ev: 0.0,
            irvec_g: vec![[0, 0, 0], [1, 0, 0]],
            u_cbm: vec![Cf::new(1.0, 0.0)],
            gbar_open: vec![vec![vec![Cf::new(2.0, 0.0)]], vec![vec![Cf::new(0.0, 1.0)]]],
        };
        let ph = Phonons {
            omega_thz: vec![5.0],
            evec: vec![vec![Cf::new(1.0, 0.0)]],
        };
        // q=1/4 ⇒ phase at R_p=(1,0,0) is e^{iπ/2}=i, so g_cart = 2 + i·i = 1.
        let g = model.g_branches(Vector3::new(0.25, 0.0, 0.0), &ph);
        assert!((g[0] - Cf::new(RY_TO_EV, 0.0)).norm() < 1e-12, "{}", g[0]);
    }
}
