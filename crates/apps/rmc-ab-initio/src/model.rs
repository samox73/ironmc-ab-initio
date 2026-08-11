//! The reduced DiagMC artifact (A1 item 8): the material-specific vertex
//! `|g(Γ,q,LO)|²` on a q-grid plus the constants Phase B needs, in one
//! serde-cached struct so a run needn't re-read the 146 MB `epmatwp`.
//!
//! The vertex is the **full** coupling — short-range from [`crate::reconstruct`]
//! plus the polar long-range from [`crate::longrange`] — projected onto the
//! optical (Fröhlich-active) branches with the phonon zero-point factor
//! `1/√(2 M_κ ω_LO)`. In the Einstein-phonon prototype `ω_LO` is held constant at
//! `ω_LO(Γ)` with the LO–TO splitting restored (the Wannier-interpolated
//! dynamical matrix has it subtracted, so we add it back in closed form). The
//! electron leg uses the reduced Γ-vertex (both legs `U(Γ)`, q-only) — the object
//! the sampler consumes; see the A2 note in `plan-rmc-ab-initio.md`.

use crate::epw::{Crystal, EpwData, Wigner};
use crate::longrange::g_long_cart;
use crate::reconstruct::{interpolate_d_polar, GammaModel, Phonons};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::path::Path;

const RY_TO_EV: f64 = 13.605698066;
const RY_TO_THZ: f64 = 3289.84498;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Self-contained reduced model: `|g(Γ,q,LO)|²` on a q-grid + material constants.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReducedModel {
    /// Fractional (crystal) q-points; Γ is excluded (the polar term diverges there).
    pub q: Vec<[f64; 3]>,
    /// `|g(Γ,q,LO)|²` per q-point, **eV²** — full coupling (short + long), summed
    /// over the optical branches, Einstein-normalized at `ω_LO(Γ)`.
    pub g2_lo: Vec<f64>,
    /// Einstein LO frequency `ω_LO(Γ)` (LO–TO splitting restored), **meV**.
    pub omega_lo_mev: f64,
    /// Split-free TO frequency `ω_TO(Γ)` (as interpolated), **meV** — for reference.
    pub omega_to_mev: f64,
    /// High-frequency dielectric constant ε∞ (isotropic scalar for LiF).
    pub eps_inf: f64,
    /// CBM energy at Γ (eV).
    pub cbm_ev: f64,
    /// Cell volume (bohr³) and lattice parameter (bohr) — for the B4 unit bridge.
    pub omega_cell: f64,
    pub alat: f64,
}

impl ReducedModel {
    /// Assemble the artifact by evaluating the full vertex on `q_points`
    /// (fractional coords, none equal to Γ).
    pub fn build(
        gamma: &GammaModel,
        epw: &EpwData,
        cry: &Crystal,
        wig: &Wigner,
        q_points: &[[f64; 3]],
    ) -> Self {
        let (omega_to_ry, omega_lo_ry) = lo_to_frequencies(epw, cry, wig);
        let g2_lo = q_points
            .iter()
            .map(|&q| {
                let short = gamma.g_cart(q);
                let long = g_long_cart(cry, epw, q);
                let total: Vec<Complex64> = (0..cry.nmodes).map(|a| short[a] + long[a]).collect();
                let ph = interpolate_d_polar(epw, cry, wig, q);
                optical_g2_ev2(&total, &ph, cry, omega_lo_ry)
            })
            .collect();

        ReducedModel {
            q: q_points.to_vec(),
            g2_lo,
            omega_lo_mev: omega_lo_ry * RY_TO_EV * 1000.0,
            omega_to_mev: omega_to_ry * RY_TO_EV * 1000.0,
            eps_inf: epw.epsi[0][0],
            cbm_ev: gamma.cbm_energy_ev,
            omega_cell: cry.omega,
            alat: cry.alat,
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }
}

/// A regular fractional q-grid over `[0,1)³`, `n` points per axis, Γ excluded.
pub fn regular_grid(n: usize) -> Vec<[f64; 3]> {
    let mut q = Vec::with_capacity(n * n * n - 1);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                if i == 0 && j == 0 && k == 0 {
                    continue;
                }
                q.push([i as f64 / n as f64, j as f64 / n as f64, k as f64 / n as f64]);
            }
        }
    }
    q
}

/// Coupling to the optical (polar-active) branches, `Σ_{ν∈opt} |g_ν|²` in eV².
/// `g_ν = Σ_a e_ν[a] · g_cart[a] / √(2 M_{a/3} ω_LO)` (Einstein: ω fixed at
/// `ω_LO(Γ)`). The two TO branches carry ≈0 polar coupling, so the sum is
/// essentially the LO channel but stays basis-independent under the degenerate
/// optical manifold.
fn optical_g2_ev2(g_cart: &[Complex64], ph: &Phonons, cry: &Crystal, omega_lo_ry: f64) -> f64 {
    let n = cry.nmodes;
    // optical = the 3 highest-|ω| branches.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| ph.omega_thz[b].abs().partial_cmp(&ph.omega_thz[a].abs()).unwrap());
    let denom = (2.0 * omega_lo_ry).sqrt();
    idx.iter()
        .take(3)
        .map(|&v| {
            let g: Complex64 = (0..n)
                .map(|a| ph.evec[v][a] * g_cart[a] / (cry.mass(a / 3)).sqrt())
                .sum();
            (g / denom).norm_sqr() * RY_TO_EV * RY_TO_EV
        })
        .sum()
}

/// Highest (LO) physical phonon frequency at fractional q, in Ry — from
/// [`interpolate_d_polar`] (short-range D + the polar dipole-dipole term), which
/// restores the LO–TO splitting at finite q. Phase B can call this per-q for the
/// dispersive-ω_LO swap.
pub fn omega_lo_ry(epw: &EpwData, cry: &Crystal, wig: &Wigner, q: [f64; 3]) -> f64 {
    interpolate_d_polar(epw, cry, wig, q).omega_thz.iter().cloned().fold(f64::MIN, f64::max) / RY_TO_THZ
}

/// `(ω_TO(Γ), ω_LO(Γ))` in Ry. TO from the physical D at exactly Γ (the non-
/// analytic G=0 term vanishes there ⇒ no splitting); LO from the q→0 limit along
/// a reference direction (the splitting is direction-independent for cubic LiF).
fn lo_to_frequencies(epw: &EpwData, cry: &Crystal, wig: &Wigner) -> (f64, f64) {
    let omega_to_ry = interpolate_d_polar(epw, cry, wig, [0.0; 3]).omega_thz.iter().cloned().fold(f64::MIN, f64::max) / RY_TO_THZ;
    (omega_to_ry, omega_lo_ry(epw, cry, wig, [1e-3, 0.0, 0.0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trip() {
        let m = ReducedModel {
            q: vec![[0.1, 0.0, 0.0], [0.2, 0.1, 0.0]],
            g2_lo: vec![1.5, 0.4],
            omega_lo_mev: 85.0,
            omega_to_mev: 49.0,
            eps_inf: 2.04,
            cbm_ev: 9.6143,
            omega_cell: 112.99,
            alat: 7.674,
        };
        let s = serde_json::to_string(&m).unwrap();
        assert_eq!(m, serde_json::from_str(&s).unwrap());
    }

    #[test]
    fn grid_excludes_gamma_and_sizes() {
        let g = regular_grid(4);
        assert_eq!(g.len(), 4 * 4 * 4 - 1);
        assert!(g.iter().all(|q| q != &[0.0, 0.0, 0.0]));
    }
}
