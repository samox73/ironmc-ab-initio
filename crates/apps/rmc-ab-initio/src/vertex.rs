//! Phase B1/B3/B4: the material vertex as an acceptance **reweight** on top of
//! the reused Fröhlich engine, plus the unit bridge to Fröhlich units.
//!
//! The Fröhlich engine samples the stationary distribution with the analytic
//! vertex `|g_F(q)|² ∝ α/q²`. Multiplying every arc-`q`-changing move's
//! acceptance by `R(q) = |g_target(q)|²/|g_F(q)|²` swaps the stationary vertex to
//! `|g_target|²` while reusing the same proposal (plan B3). For a **radial**
//! reweight only Add / Remove / ChangeQModulus need it — ChangeQDirection and
//! ChangeTopology preserve the arc-`|q|` multiset. See [`crate::updates`].
//!
//! Two targets:
//! - [`Reweight::constant`] — the Fröhlich limit, `R ≡ α_target/α_engine`
//!   (momentum-independent; both `1/q²`). At `α_target = α_engine` this is
//!   `R ≡ 1` and the run is bit-for-bit the Fröhlich engine.
//! - [`radial_from_model`] — the real LiF tensor `|g_ab(|q|)|²` (spherically
//!   averaged), normalized at small `q` to the material Fröhlich coupling
//!   [`frohlich_alpha`]. Its physical validation is the stage-2 run (a USER
//!   CHECKPOINT — see `plan-rmc-ab-initio.md`).

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::model::ReducedModel;
use nalgebra::Vector3;

/// ħ²/m_e in eV·Å² (so ħ²/2m_e = 3.80998 eV·Å²).
const HBAR2_OVER_ME: f64 = 7.6199682;
/// e²/(4πε₀) in eV·Å.
const E2_OVER_4PI_EPS0: f64 = 14.399645;
/// Bohr radius in Å (crystal `alat` is stored in bohr).
const BOHR_TO_ANG: f64 = 0.529177210903;

/// Material constants for the Fröhlich unit bridge (plan B4). `mass_eff` is
/// `m*/m_e`; `eps_zero`/`eps_inf` the static/high-frequency dielectric constants.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Material {
    pub eps_inf: f64,
    pub eps_zero: f64,
    pub omega_lo_mev: f64,
    pub mass_eff: f64,
}

impl Material {
    /// Polaron length `r_p = √(ħ/(2 m* ω_LO))` in Å. Also the inverse of the
    /// Fröhlich wavevector unit: a Fröhlich momentum `q_F` maps to the physical
    /// wavevector `k = q_F / r_p`, i.e. `q_F = k·r_p`.
    pub fn polaron_length_ang(&self) -> f64 {
        let hw_ev = self.omega_lo_mev / 1000.0;
        (HBAR2_OVER_ME / 2.0 / (self.mass_eff * hw_ev)).sqrt()
    }

    /// Fröhlich coupling `α` from `(ε∞, ε0, ω_LO, m*)` (plan B4). Textbook
    /// `α = (1/ε∞ − 1/ε0)·(e²/4πε₀)·√(m*/(2ħω_LO))/ħ`.
    pub fn frohlich_alpha(&self) -> f64 {
        let ionicity = 1.0 / self.eps_inf - 1.0 / self.eps_zero;
        ionicity * self.mass_eff * (E2_OVER_4PI_EPS0 / HBAR2_OVER_ME) * self.polaron_length_ang()
    }

    /// Ground-state energy back to meV: `E[meV] = E_Fröhlich · ħω_LO[meV]`.
    pub fn energy_to_mev(&self, e_frohlich: f64) -> f64 {
        e_frohlich * self.omega_lo_mev
    }
}

/// A momentum-space acceptance reweight `R(q)`, shared cheaply across the updates.
#[derive(Clone, Debug)]
pub struct Reweight(Arc<Kind>);

#[derive(Debug)]
enum Kind {
    /// `R ≡ 1`: pure Fröhlich, bit-for-bit.
    Identity,
    /// `R ≡ c`: momentum-independent (Fröhlich limit, `α_target/α_engine`).
    Constant(f64),
    /// `R(|q|)`: linear interpolation over sorted Fröhlich-unit `|q|` nodes.
    Radial { q: Vec<f64>, r: Vec<f64> },
}

impl Default for Reweight {
    fn default() -> Self {
        Self::identity()
    }
}

impl Reweight {
    pub fn identity() -> Self {
        Reweight(Arc::new(Kind::Identity))
    }

    /// Fröhlich-limit reweight `α_target/α_engine` (momentum-independent).
    pub fn constant(c: f64) -> Self {
        Reweight(Arc::new(Kind::Constant(c)))
    }

    /// Tabulated radial reweight. `q` sorted ascending, same length as `r`, non-empty.
    pub fn radial(q: Vec<f64>, r: Vec<f64>) -> Self {
        assert_eq!(q.len(), r.len());
        assert!(!q.is_empty());
        assert!(
            q.windows(2).all(|w| w[0] <= w[1]),
            "radial |q| nodes must be sorted"
        );
        Reweight(Arc::new(Kind::Radial { q, r }))
    }

    pub fn is_identity(&self) -> bool {
        matches!(*self.0, Kind::Identity)
    }

    /// `R(q)`. Identity returns exactly `1.0` (so `ratio * factor == ratio`).
    pub fn factor(&self, q: Vector3<f64>) -> f64 {
        match &*self.0 {
            Kind::Identity => 1.0,
            Kind::Constant(c) => *c,
            Kind::Radial { q: nodes, r } => interp_clamped(nodes, r, q.norm()),
        }
    }
}

/// Linear interpolation of `r` over sorted `nodes` at `x`, clamped to the end
/// values outside `[nodes.first, nodes.last]`.
fn interp_clamped(nodes: &[f64], r: &[f64], x: f64) -> f64 {
    if x <= nodes[0] {
        return r[0];
    }
    let last = nodes.len() - 1;
    if x >= nodes[last] {
        return r[last];
    }
    let hi = nodes.partition_point(|&n| n <= x).min(last);
    let lo = hi - 1;
    let w = (x - nodes[lo]) / (nodes[hi] - nodes[lo]);
    r[lo] + w * (r[hi] - r[lo])
}

/// Build the radial reweight `R(|q|)` from a reduced LiF model.
///
/// The tensor gives the q-**shape**; the material Fröhlich `α` gives the small-q
/// **normalization** (the standard ab-initio Fröhlich check: `|g_ab|²·q² → C·α`
/// as `q→0`). Concretely, per `|q|`-shell we average `|g_ab|²`, form the local
/// effective coupling `α_eff(q) = α_LiF · (⟨g²⟩·q²)/(⟨g²⟩·q²)|_ref` with the
/// reference the smallest-|q| shell, and tabulate `R = α_eff/α_engine`.
///
/// ponytail: `|q|` uses the cubic-isotropic map `|k| = (2π/alat)·|q_frac|` and a
/// spherical average — matches the prototype's near-isotropic `g(k,q)≈g(Γ,q)`
/// scope. Full FCC reciprocal geometry + anisotropic (3-D) `R(q)` is the deferred
/// upgrade; physical accuracy is the stage-2 checkpoint.
pub fn radial_from_model(
    reduced: &ReducedModel,
    material: &Material,
    alpha_engine: f64,
) -> Reweight {
    let alpha_lif = material.frohlich_alpha();
    let alat_ang = reduced.alat * BOHR_TO_ANG;
    let k_per_frac = 2.0 * std::f64::consts::PI / alat_ang; // Å⁻¹ per unit |q_frac|
    let r_p = material.polaron_length_ang();

    // Spherically average |g|² over |q_frac| shells (keyed to ~1e-6 resolution).
    let mut shells: BTreeMap<u64, (f64, usize)> = BTreeMap::new();
    for (qf, &g2) in reduced.q.iter().zip(&reduced.g2_lo) {
        let qabs = qf.norm();
        let key = (qabs * 1e6).round() as u64;
        let e = shells.entry(key).or_insert((0.0, 0));
        e.0 += g2;
        e.1 += 1;
    }

    // Reference plateau: smallest-|q| shell (deepest in the 1/q² region).
    let mut rows: Vec<(f64, f64, f64)> = shells // (|q_frac|, ⟨g²⟩, g²·q²)
        .iter()
        .map(|(&key, &(sum, n))| {
            let qabs = key as f64 / 1e6;
            let g2 = sum / n as f64;
            (qabs, g2, g2 * qabs * qabs)
        })
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let plateau = rows[0].2;

    let (mut q_nodes, mut r_nodes) = (
        Vec::with_capacity(rows.len()),
        Vec::with_capacity(rows.len()),
    );
    for (qabs, _g2, g2q2) in rows {
        let alpha_eff = alpha_lif * g2q2 / plateau;
        let q_frohlich = qabs * k_per_frac * r_p;
        q_nodes.push(q_frohlich);
        r_nodes.push(alpha_eff / alpha_engine);
    }
    Reweight::radial(q_nodes, r_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_constant() {
        assert_eq!(
            Reweight::identity().factor(Vector3::new(0.3, 0.4, 0.0)),
            1.0
        );
        assert!(Reweight::identity().is_identity());
        assert_eq!(
            Reweight::constant(2.5).factor(Vector3::new(9.0, 0.0, 0.0)),
            2.5
        );
    }

    #[test]
    fn radial_interpolates_clamps_and_is_isotropic() {
        let rw = Reweight::radial(vec![1.0, 2.0, 4.0], vec![10.0, 20.0, 40.0]);
        assert!((rw.factor(Vector3::new(1.5, 0.0, 0.0)) - 15.0).abs() < 1e-12);
        assert!((rw.factor(Vector3::new(0.0, 1.5, 0.0)) - 15.0).abs() < 1e-12);
        assert_eq!(rw.factor(Vector3::new(0.5, 0.0, 0.0)), 10.0);
        assert_eq!(rw.factor(Vector3::new(9.0, 0.0, 0.0)), 40.0);
    }

    #[test]
    fn lif_alpha_is_about_five() {
        // LiF: ε∞≈2.04, ε0≈9.0, ω_LO≈77 meV, m*≈0.9 m_e → α≈4.8 (plan: α≈5).
        let m = Material {
            eps_inf: 2.04,
            eps_zero: 9.0,
            omega_lo_mev: 77.0,
            mass_eff: 0.9,
        };
        let a = m.frohlich_alpha();
        assert!((4.0..6.0).contains(&a), "α_LiF = {a}");
    }

    #[test]
    fn synthetic_frohlich_tensor_gives_flat_alpha() {
        // A pure 1/q² model (g² = C/q_frac²) must yield α_eff ≡ α_LiF, so with
        // α_engine = α_LiF the radial reweight is ≈ 1 at every node.
        let m = Material {
            eps_inf: 2.04,
            eps_zero: 9.0,
            omega_lo_mev: 77.0,
            mass_eff: 0.9,
        };
        let q: Vec<Vector3<f64>> = (1..=8)
            .map(|i| Vector3::new(i as f64 * 0.05, 0.0, 0.0))
            .collect();
        let g2_lo: Vec<f64> = q.iter().map(|v| 1.0 / (v[0] * v[0])).collect();
        let reduced = ReducedModel {
            q,
            g2_lo,
            omega_lo_mev: m.omega_lo_mev,
            omega_to_mev: 33.0,
            eps_inf: m.eps_inf,
            cbm_ev: 9.6,
            omega_cell: 113.0,
            alat: 7.674,
        };
        let rw = radial_from_model(&reduced, &m, m.frohlich_alpha());
        // Every shell's α_eff should equal α_LiF ⇒ R ≈ 1 at every node.
        if let Kind::Radial { r, .. } = &*rw.0 {
            for &ri in r {
                assert!(
                    (ri - 1.0).abs() < 1e-9,
                    "R = {ri} (expected 1 for 1/q² tensor)"
                );
            }
        } else {
            panic!("expected radial");
        }
    }
}
