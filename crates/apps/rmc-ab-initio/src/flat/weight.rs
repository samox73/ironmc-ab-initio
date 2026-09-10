//! Local weight ratios on the flat diagram — a line-by-line port of
//! `rmc_diagmc::weight` onto u32 grid momenta and the index-level
//! `AbInitioModel` accessors. Window semantics (half-open endpoint tests, the
//! skip at `key == right`) must stay identical to the slotmap versions; the
//! parity suite pins them.

use num_complex::Complex64;

use super::{neg, wrap_sub, FlatDiagram, NONE};
use crate::model_abinitio::AbInitioModel;
use rmc_diagmc::weight::LocalRatio;

/// One arc's vertex convention: `g(k_emit,q,ν) · g(k_absorb,q,ν)*`.
#[inline]
fn arc_factor(model: &AbInitioModel, k_emit: u32, k_absorb: u32, q: u32, nu: u32) -> Complex64 {
    model.vertex_index(k_emit as usize, q as usize, nu as usize)
        * model
            .vertex_index(k_absorb as usize, q as usize, nu as usize)
            .conj()
}

pub fn insert_arc_ratio(
    d: &FlatDiagram,
    model: &AbInitioModel,
    tau1: f64,
    tau2: f64,
    q: u32,
    branch: u32,
) -> LocalRatio {
    let n = d.grid_n;
    let changed_arcs = changed_arc_factors(d, model, tau1, tau2, q, None);
    let mut log_modulus =
        segment_change(d, model, tau1, tau2, |p| wrap_sub(n, p, q)) + changed_arcs.log_modulus;
    let first = d.segment_at(tau1);
    let last = d.segment_at(tau2);
    let factor = arc_factor(
        model,
        wrap_sub(n, d.p_out[first], q),
        wrap_sub(n, d.p_out[last], q),
        q,
        branch,
    );
    let modulus = factor.norm();
    if modulus == 0.0 || !modulus.is_finite() {
        return LocalRatio {
            log_modulus: f64::NEG_INFINITY,
            phase: Complex64::new(1.0, 0.0),
        };
    }
    log_modulus += modulus.ln()
        - model.ln_nq()
        - model.phonon_ev_index(q as usize, branch as usize) * (tau2 - tau1);
    LocalRatio {
        log_modulus,
        phase: changed_arcs.phase * factor / modulus,
    }
}

pub fn remove_arc_ratio(
    d: &FlatDiagram,
    model: &AbInitioModel,
    left: usize,
    right: usize,
) -> LocalRatio {
    debug_assert_eq!(d.link[left] as usize, right);
    let n = d.grid_n;
    let q = d.q[left];
    let branch = d.branch[left];
    let factor = arc_factor(model, d.p_out[left], d.incoming_momentum(right), q, branch);
    let modulus = factor.norm();
    if modulus == 0.0 || !modulus.is_finite() {
        return LocalRatio {
            log_modulus: f64::INFINITY,
            phase: Complex64::new(1.0, 0.0),
        };
    }
    let changed_arcs = changed_arc_factors(
        d,
        model,
        d.tau[left],
        d.tau[right],
        neg(n, q),
        Some((left, right)),
    );
    let mut log_modulus = segment_change(d, model, d.tau[left], d.tau[right], |p| {
        super::wrap_add(n, p, q)
    });
    log_modulus += changed_arcs.log_modulus - modulus.ln()
        + model.ln_nq()
        + model.phonon_ev_index(q as usize, branch as usize) * (d.tau[right] - d.tau[left]);
    LocalRatio {
        log_modulus,
        phase: changed_arcs.phase * (factor / modulus).conj(),
    }
}

pub fn shift_vertex_tau_ratio(
    d: &FlatDiagram,
    model: &AbInitioModel,
    key: usize,
    new_tau: f64,
) -> LocalRatio {
    debug_assert!(key > 0 && key < d.tail());
    let delta = new_tau - d.tau[key];
    let before = model.electron_ev_index(d.p_out[key - 1] as usize);
    let after = model.electron_ev_index(d.p_out[key] as usize);
    let omega = model.phonon_ev_index(d.q[key] as usize, d.branch[key] as usize);
    let phonon = omega * if (d.link[key] as usize) > key { delta } else { -delta };
    LocalRatio {
        log_modulus: (after - before) * delta + phonon,
        phase: Complex64::new(1.0, 0.0),
    }
}

pub fn shift_tail_ratio(d: &FlatDiagram, model: &AbInitioModel, new_tau: f64) -> LocalRatio {
    let delta = new_tau - d.tau_ext();
    LocalRatio {
        log_modulus: -(model.electron_ev_index(d.p_out[d.tail() - 1] as usize) - d.mu) * delta,
        phase: Complex64::new(1.0, 0.0),
    }
}

fn segment_change(
    d: &FlatDiagram,
    model: &AbInitioModel,
    tau1: f64,
    tau2: f64,
    changed: impl Fn(u32) -> u32,
) -> f64 {
    let mut key = d.segment_at(tau1);
    let mut out = 0.0;
    while key < d.tail() && d.tau[key] < tau2 {
        let left = tau1.max(d.tau[key]);
        let right = tau2.min(d.tau[key + 1]);
        if right > left {
            let p = d.p_out[key];
            out -= (model.electron_ev_index(changed(p) as usize)
                - model.electron_ev_index(p as usize))
                * (right - left);
        }
        key += 1;
    }
    out
}

fn changed_arc_factors(
    d: &FlatDiagram,
    model: &AbInitioModel,
    tau1: f64,
    tau2: f64,
    inserted_q: u32,
    skip: Option<(usize, usize)>,
) -> LocalRatio {
    let n = d.grid_n;
    let mut ratio = LocalRatio {
        log_modulus: 0.0,
        phase: Complex64::new(1.0, 0.0),
    };
    let start = d.tau[..d.len].partition_point(|&t| t < tau1);
    let end = d.tau[..d.len].partition_point(|&t| t <= tau2).min(d.tail());
    for key in start..end {
        let linked = d.link[key];
        if linked == NONE {
            continue;
        }
        let linked = linked as usize;
        let (left, right) = if key < linked { (key, linked) } else { (linked, key) };
        let old_emit = d.p_out[left];
        let old_absorb = d.incoming_momentum(right);
        let emit_changed = d.tau[left] >= tau1 && d.tau[left] < tau2;
        let absorb_changed = d.tau[right] > tau1 && d.tau[right] <= tau2;
        if skip == Some((left, right))
            || (!emit_changed && !absorb_changed)
            || (key == right && emit_changed)
        {
            continue;
        }
        let q = d.q[left];
        let nu = d.branch[left];
        let old = arc_factor(model, old_emit, old_absorb, q, nu);
        let new_emit = if emit_changed { wrap_sub(n, old_emit, inserted_q) } else { old_emit };
        let new_absorb = if absorb_changed {
            wrap_sub(n, old_absorb, inserted_q)
        } else {
            old_absorb
        };
        let new = arc_factor(model, new_emit, new_absorb, q, nu);
        let old_modulus = old.norm();
        let new_modulus = new.norm();
        if old_modulus == 0.0 || new_modulus == 0.0 {
            ratio.log_modulus = if new_modulus == 0.0 { f64::NEG_INFINITY } else { f64::INFINITY };
            ratio.phase = Complex64::new(1.0, 0.0);
            return ratio;
        }
        ratio.log_modulus += new_modulus.ln() - old_modulus.ln();
        ratio.phase *= new / new_modulus * (old / old_modulus).conj();
    }
    ratio
}

/// Full recompute, mirroring `weight::log_modulus_and_phase` (drift checks and
/// cache initialization).
pub fn log_modulus_and_phase(d: &FlatDiagram, model: &AbInitioModel) -> (f64, Complex64) {
    let mut log_modulus = 0.0;
    let mut total_phase = Complex64::new(1.0, 0.0);

    for key in 0..d.tail() {
        let dt = d.tau[key + 1] - d.tau[key];
        log_modulus -= (model.electron_ev_index(d.p_out[key] as usize) - d.mu) * dt;
    }

    for (left, right) in d.arcs() {
        let q = d.q[left];
        let nu = d.branch[left];
        let factor = arc_factor(model, d.p_out[left], d.incoming_momentum(right), q, nu);
        let modulus = factor.norm();
        if modulus == 0.0 || !modulus.is_finite() {
            return (f64::NEG_INFINITY, Complex64::new(1.0, 0.0));
        }
        log_modulus += modulus.ln()
            - model.ln_nq()
            - model.phonon_ev_index(q as usize, nu as usize) * (d.tau[right] - d.tau[left]);
        total_phase *= factor / modulus;
    }

    let norm = total_phase.norm();
    if norm > 0.0 {
        total_phase /= norm;
    }
    (log_modulus, total_phase)
}

pub fn sync_cache(d: &mut FlatDiagram, model: &AbInitioModel) {
    let (log_modulus, phase) = log_modulus_and_phase(d, model);
    d.set_cached_weight(log_modulus, phase);
}
