//! CubeCL Monte Carlo step kernel: one Fröhlich chain per GPU thread.
//!
//! The kernel is a line-by-line port of `flat::updates` + `flat::batched::step_groups`. Every RNG
//! draw maps 1:1 onto the CPU `PhiloxRng` word stream keyed by `(seed, chain, step)`, and update
//! selection is uniform per cube (workgroup), keyed by `(seed, CUBE_POS_X, step)` exactly like
//! `group_update_index`. Per cycle each thread records `(tau, order, exact_estimator)`; all
//! statistics stay on the host (`PolaronMeasurement::measure_flat_sample`).
//!
//! Physics constants are folded at `MASS = OMEGA = 1` (asserted in `launch_segment`).
//!
//! ponytail: everything is inlined in one mega-kernel with scalar-only helpers — the only cube
//! constructs used are the ones the Phase-0 spike validated. Extract shared walks into
//! array-taking cube fns once a second model needs them.

use cubecl::prelude::*;

use crate::config::RunConfig;
use crate::flat::batched::{run_batched, BatchedRunOutput};

pub const DEFAULT_WORKGROUP_SIZE: u32 = 64;

/// Phase-2 CPU reference rig with identical batched-chain semantics; kept as the parity oracle.
pub fn launch_reference_kernel(cfg: &RunConfig) -> rmc_core::Result<BatchedRunOutput> {
    run_batched(cfg, cfg.chains as usize, DEFAULT_WORKGROUP_SIZE as usize)
}

/// `flat::NULL` as stored in the u32 state arrays.
const NULL: u32 = 0xffff_ffff;
/// `flat::NULL` widened for usize-typed slot locals.
const NULLS: usize = 0xffff_ffff;
/// 2^-53, the scale of `rand`'s 53-bit `f64` conversion.
const U01_SCALE: f64 = 1.110_223_024_625_156_5e-16;
/// `f64::EPSILON`.
const EPS: f64 = 2.220_446_049_250_313e-16;
/// `physics::DELTA_TAU_LIMIT` = 10 * f64::EPSILON.
const DELTA_TAU_LIMIT: f64 = 2.220_446_049_250_313e-15;
/// `physics::p0()` = sqrt(2 * MASS * OMEGA) = sqrt(2).
const P0: f64 = 1.414_213_562_373_095_1;
/// `f64::MAX`, for the finiteness test (false for NaN and infinities).
const F64_MAX: f64 = 1.797_693_134_862_315_7e308;
const PI: f64 = core::f64::consts::PI;

// ---------------------------------------------------------------------------
// Philox4x32-10 counter stream (mirrors flat::philox exactly)
// ---------------------------------------------------------------------------

#[cube]
fn mul_hi(a: u32, b: u32) -> u32 {
    ((u64::cast_from(a) * u64::cast_from(b)) >> 32) as u32
}

#[cube]
fn mul_lo(a: u32, b: u32) -> u32 {
    (u64::cast_from(a) * u64::cast_from(b)) as u32
}

/// Word `select` of `philox4x32_10([c0, c1, c2, c3], [seed_lo, seed_hi])`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn philox_block_word(
    seed_lo: u32,
    seed_hi: u32,
    mut c0: u32,
    mut c1: u32,
    mut c2: u32,
    mut c3: u32,
    select: u32,
) -> u32 {
    let mut k0 = seed_lo;
    let mut k1 = seed_hi;
    #[unroll]
    for round in 0..10 {
        if comptime![round > 0] {
            k0 += 0x9e37_79b9u32;
            k1 += 0xbb67_ae85u32;
        }
        let hi0 = mul_hi(0xd251_1f53u32, c0);
        let lo0 = mul_lo(0xd251_1f53u32, c0);
        let hi1 = mul_hi(0xcd9e_8d57u32, c2);
        let lo1 = mul_lo(0xcd9e_8d57u32, c2);
        let n0 = hi1 ^ c1 ^ k0;
        let n1 = lo1;
        let n2 = hi0 ^ c3 ^ k1;
        let n3 = lo0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
    }
    let mut out = c0;
    if select == 1 {
        out = c1;
    }
    if select == 2 {
        out = c2;
    }
    if select == 3 {
        out = c3;
    }
    out
}

/// Word `word` of the CPU `PhiloxRng { seed, chain_id: id, step }` stream:
/// word `word % 4` of the block with counter `[id_lo, id_hi, step_lo, step_hi ^ (word / 4)]`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn philox_word(
    seed_lo: u32,
    seed_hi: u32,
    id_lo: u32,
    id_hi: u32,
    step_lo: u32,
    step_hi: u32,
    word: u32,
) -> u32 {
    philox_block_word(
        seed_lo,
        seed_hi,
        id_lo,
        id_hi,
        step_lo,
        step_hi ^ (word / 4),
        word % 4,
    )
}

/// `rand`'s `gen::<f64>()` on the CPU Philox stream: words `word`, `word + 1` assembled
/// low-first into a u64, then `(v >> 11) * 2^-53`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn u01(
    seed_lo: u32,
    seed_hi: u32,
    id_lo: u32,
    id_hi: u32,
    step_lo: u32,
    step_hi: u32,
    word: u32,
) -> f64 {
    let lo = philox_word(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, word);
    let hi = philox_word(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, word + 1);
    let bits = u64::cast_from(lo) | (u64::cast_from(hi) << 32);
    f64::cast_from(bits >> 11) * U01_SCALE
}

// ---------------------------------------------------------------------------
// Portable samplers (mirror rmc_core::random::samples)
// ---------------------------------------------------------------------------

/// Finite (neither NaN nor infinite): NaN fails any comparison, infinities exceed `F64_MAX`.
#[cube]
fn is_finite_f64(x: f64) -> bool {
    x.abs() <= F64_MAX
}

/// `exp(x) - 1` with a short series below |x| < 1e-4 (no expm1 in the cube dialect).
#[cube]
fn exp_m1(x: f64) -> f64 {
    if x.abs() < 1.0e-4 {
        x * (1.0 + x * (0.5 + x * (1.0 / 6.0)))
    } else {
        x.exp() - 1.0
    }
}

#[cube]
fn exponential_sample_bounded(r: f64, lambda: f64, a: f64, b: f64) -> f64 {
    a - (r * exp_m1(-(lambda * (b - a)))).log1p() / lambda
}

#[cube]
fn safe_exponential_sample(r: f64, lambda: f64, a: f64, b: f64) -> f64 {
    if lambda > 0.0 {
        exponential_sample_bounded(r, lambda, a, b)
    } else if lambda < 0.0 {
        b - exponential_sample_bounded(r, -lambda, 0.0, b - a)
    } else {
        a + r * (b - a)
    }
}

#[cube]
fn normal_from_uniforms(r1: f64, r2: f64, mean: f64, sigma: f64) -> f64 {
    let radius = (-2.0 * (-r1).log1p()).sqrt();
    mean + sigma * radius * (2.0 * PI * r2).cos()
}

#[cube]
fn uniform_index(r: f64, len: u32) -> u32 {
    let idx = (r * f64::cast_from(len)) as u32;
    if idx >= len {
        len - 1
    } else {
        idx
    }
}

// ---------------------------------------------------------------------------
// Physics leaves (mirror physics.rs at MASS = OMEGA = 1)
// ---------------------------------------------------------------------------

#[cube]
fn dot3(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    ax * bx + ay * by + az * bz
}

#[cube]
fn norm_sq3(x: f64, y: f64, z: f64) -> f64 {
    x * x + y * y + z * z
}

#[cube]
fn norm3(x: f64, y: f64, z: f64) -> f64 {
    norm_sq3(x, y, z).sqrt()
}

#[cube]
fn bare_dispersion(px: f64, py: f64, pz: f64) -> f64 {
    norm_sq3(px, py, pz) / 2.0
}

#[cube]
fn dispersion(px: f64, py: f64, pz: f64, mu: f64) -> f64 {
    bare_dispersion(px, py, pz) - mu
}

#[cube]
fn phonon_lambda(qx: f64, qy: f64, qz: f64) -> f64 {
    let t = 1.0 + norm3(qx, qy, qz) / P0;
    t * t
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn change_internal_tau_lambda(
    px: f64,
    py: f64,
    pz: f64,
    cx: f64,
    cy: f64,
    cz: f64,
    incoming: bool,
) -> f64 {
    // NOTE: if-expressions whose branches are both literals miscompile in the cube macro
    // (the runtime condition is ignored); use statement-form ifs for literal selects.
    let mut omega = 1.0f64;
    if !incoming {
        omega = -omega;
    }
    bare_dispersion(px, py, pz) - bare_dispersion(cx, cy, cz) + omega
}

// ---------------------------------------------------------------------------
// The MC kernel
// ---------------------------------------------------------------------------

/// Steps every chain `n_cycles` cycles forward and records one sample per cycle.
///
/// State layout: slot `s` of chain `c` lives at `c * capacity + s`; `p_out` and `q` are flattened
/// `[x, y, z]` triples at `3 * (c * capacity + s) + component`. Samples are cycle-major, update
/// stats chain-major.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn mc_kernel(
    tau: &mut Array<f64>,
    p_out: &mut Array<f64>,
    q: &mut Array<f64>,
    link: &mut Array<u32>,
    prev: &mut Array<u32>,
    next: &mut Array<u32>,
    storage_idx: &mut Array<u32>,
    phonons_above: &mut Array<u32>,
    storage: &mut Array<u32>,
    storage_len: &mut Array<u32>,
    head: &mut Array<u32>,
    tail: &mut Array<u32>,
    order: &mut Array<u32>,
    samples_tau: &mut Array<f64>,
    samples_exact: &mut Array<f64>,
    samples_order: &mut Array<u32>,
    stats_proposed: &mut Array<u32>,
    stats_accepted: &mut Array<u32>,
    stats_impossible: &mut Array<u32>,
    seed_lo: u32,
    seed_hi: u32,
    step0: u64,
    n_chains: u32,
    capacity: u32,
    n_cycles: u32,
    steps_per_cycle: u32,
    last_cycle_steps: u32,
    alpha: f64,
    mu: f64,
    max_tau: f64,
    min_order: u32,
    max_order: u32,
    num_bins: u32,
) {
    let chain = ABSOLUTE_POS;
    if chain < n_chains as usize {
        // ponytail: chain-major layout; flip to slot-major here if real-GPU profiling says so.
        let base = chain * capacity as usize;
        let group = CUBE_POS_X;
        let id_lo = chain as u32;
        let id_hi = 0u32;

        for cycle in 0..n_cycles {
            let steps = if cycle + 1 == n_cycles {
                last_cycle_steps
            } else {
                steps_per_cycle
            };
            for s in 0..steps {
                let step = step0
                    + u64::cast_from(cycle) * u64::cast_from(steps_per_cycle)
                    + u64::cast_from(s);
                let step_lo = step as u32;
                let step_hi = (step >> 32) as u32;

                // Group-uniform update selection: word 0 of the (seed, group, step) block with
                // counter word 3 = step_hi ^ u32::MAX, exactly `group_update_index`.
                let upd = philox_block_word(
                    seed_lo,
                    seed_hi,
                    group,
                    0u32,
                    step_lo,
                    step_hi ^ 0xffff_ffffu32,
                    0u32,
                ) % 8;

                // Per-chain proposal stream for this step.
                let mut k = 0u32; // next unread u32 word
                let mut probability = 0.0f64;
                let mut accepted = false;

                // Proposal registers shared across the update branches.
                // Literal inits (not the NULLS const path) so the macro makes them runtime vars.
                let mut v1 = 0xffff_ffffusize;
                let mut v2 = 0xffff_ffffusize;
                let mut f1 = 0.0f64; // tau_prime | tau1 | q_prime modulus | topology flag
                let mut f2 = 0.0f64; // tau2 | topology flag
                let mut qx = 0.0f64; // proposed q | p_prime (topology)
                let mut qy = 0.0f64;
                let mut qz = 0.0f64;

                let tl = tail[chain] as usize;
                let hd = head[chain] as usize;
                let ord = order[chain];
                let n_vertices = storage_len[chain];
                let tau_total = tau[base + tl];
                let pout_x = p_out[3 * (base + tl)];
                let pout_y = p_out[3 * (base + tl) + 1];
                let pout_z = p_out[3 * (base + tl) + 2];

                if upd == 0 {
                    // --- ChangeTau ---
                    let second_last = prev[base + tl] as usize;
                    let i = base + second_last;
                    // literal-select via statement-if; see change_internal_tau_lambda note
                    let mut extra = 0.0f64;
                    if ord != 0 {
                        extra = 1.0;
                    }
                    let lambda =
                        dispersion(p_out[3 * i], p_out[3 * i + 1], p_out[3 * i + 2], mu) + extra;
                    let r = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                    k += 2;
                    f1 = safe_exponential_sample(r, lambda, tau[i], max_tau);
                    v1 = tl;
                    // tau-degeneracy guard: strict ordering is load-bearing (see flat::updates)
                    if f1 - tau[i] >= DELTA_TAU_LIMIT {
                        probability = 1.0;
                    }
                } else if upd == 1 {
                    // --- ChangeInternalTau ---
                    if ord > 1 {
                        let r0 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                        k += 2;
                        let mut v = storage[base + uniform_index(r0, n_vertices) as usize] as usize;
                        while v == hd || v == tl {
                            let r = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                            k += 2;
                            v = storage[base + uniform_index(r, n_vertices) as usize] as usize;
                        }
                        let pv = prev[base + v] as usize;
                        let nx = next[base + v];
                        let tau_previous = tau[base + pv];
                        let tau_next = if nx == NULL {
                            max_tau
                        } else {
                            tau[base + nx as usize]
                        };
                        let lk = link[base + v];
                        let incoming = lk != NULL && tau[base + lk as usize] < tau[base + v];
                        let lambda = change_internal_tau_lambda(
                            p_out[3 * (base + pv)],
                            p_out[3 * (base + pv) + 1],
                            p_out[3 * (base + pv) + 2],
                            p_out[3 * (base + v)],
                            p_out[3 * (base + v) + 1],
                            p_out[3 * (base + v) + 2],
                            incoming,
                        );
                        let r = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                        k += 2;
                        f1 = safe_exponential_sample(r, lambda, tau_previous, tau_next);
                        v1 = v;
                        // tau-degeneracy guard (see flat::updates)
                        if is_finite_f64(f1)
                            && f1 - tau_previous >= DELTA_TAU_LIMIT
                            && tau_next - f1 >= DELTA_TAU_LIMIT
                        {
                            probability = 1.0;
                        }
                    }
                } else if upd == 2 {
                    // --- AddPhonon ---
                    if ord == 0 {
                        let r1 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                        let r2 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k + 2);
                        let r3 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k + 4);
                        k += 6;
                        let theta = (1.0 - 2.0 * r1).acos();
                        let qn = P0 / r2 - P0;
                        let phi = 2.0 * PI * r3;
                        qx = qn * phi.cos() * theta.sin();
                        qy = qn * phi.sin() * theta.sin();
                        qz = qn * theta.cos();
                        // add_phonon_zero_ratio
                        let e = 1.0
                            + (norm_sq3(qx, qy, qz) / 2.0
                                - dot3(qx, qy, qz, pout_x, pout_y, pout_z));
                        let t = 1.0 + norm3(qx, qy, qz) / P0;
                        probability = 2.0 * alpha / PI * (-(e * tau_total)).exp() * t * t;
                    } else if ord >= max_order {
                        probability = 0.0;
                    } else if n_vertices + 2 > capacity {
                        probability = -1.0;
                    } else {
                        let r0 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                        k += 2;
                        let mut va =
                            storage[base + uniform_index(r0, n_vertices) as usize] as usize;
                        while next[base + va] == NULL {
                            let r = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                            k += 2;
                            va = storage[base + uniform_index(r, n_vertices) as usize] as usize;
                        }
                        let next1 = next[base + va] as usize;
                        let delta_t = tau[base + next1] - tau[base + va];
                        let rt = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                        k += 2;
                        let tau1 = tau[base + va] + rt * delta_t;
                        if tau1 - tau[base + va] < DELTA_TAU_LIMIT
                            || tau[base + next1] - tau1 < DELTA_TAU_LIMIT
                        {
                            probability = 0.0;
                        } else {
                            let r1 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                            let r2 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k + 2);
                            let r3 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k + 4);
                            k += 6;
                            let theta = (1.0 - 2.0 * r1).acos();
                            let qn = P0 / r2 - P0;
                            let phi = 2.0 * PI * r3;
                            qx = qn * phi.cos() * theta.sin();
                            qy = qn * phi.sin() * theta.sin();
                            qz = qn * theta.cos();
                            let lambda = phonon_lambda(qx, qy, qz);
                            let re = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                            k += 2;
                            let tau2 = exponential_sample_bounded(re, lambda, tau1, max_tau);
                            if tau2 - tau1 < DELTA_TAU_LIMIT {
                                probability = 0.0;
                            } else {
                                // get_p_mean_between(tau1, tau2, va)
                                let mut end = next[base + va] as usize;
                                let mut pmx = p_out[3 * (base + va)];
                                let mut pmy = p_out[3 * (base + va) + 1];
                                let mut pmz = p_out[3 * (base + va) + 2];
                                if end != NULLS && tau[base + end] < tau2 {
                                    let mut it = va;
                                    pmx = p_out[3 * (base + it)] * (tau[base + end] - tau1);
                                    pmy = p_out[3 * (base + it) + 1] * (tau[base + end] - tau1);
                                    pmz = p_out[3 * (base + it) + 2] * (tau[base + end] - tau1);
                                    it = end;
                                    end = next[base + end] as usize;
                                    while end != NULLS && tau[base + end] < tau2 {
                                        let dt = tau[base + end] - tau[base + it];
                                        pmx += p_out[3 * (base + it)] * dt;
                                        pmy += p_out[3 * (base + it) + 1] * dt;
                                        pmz += p_out[3 * (base + it) + 2] * dt;
                                        it = end;
                                        end = next[base + end] as usize;
                                    }
                                    let dt = tau2 - tau[base + it];
                                    pmx += p_out[3 * (base + it)] * dt;
                                    pmy += p_out[3 * (base + it) + 1] * dt;
                                    pmz += p_out[3 * (base + it) + 2] * dt;
                                    let inv = 1.0 / (tau2 - tau1);
                                    pmx *= inv;
                                    pmy *= inv;
                                    pmz *= inv;
                                }
                                let vb = end;
                                let prev2 = if vb == NULLS {
                                    tl
                                } else {
                                    prev[base + vb] as usize
                                };
                                let prev_tau = tau[base + prev2];
                                let far_ok =
                                    vb == NULLS || tau[base + vb] - tau2 >= DELTA_TAU_LIMIT;
                                if tau2 - prev_tau < DELTA_TAU_LIMIT || !far_ok {
                                    probability = 0.0;
                                } else {
                                    v1 = va;
                                    v2 = vb;
                                    f1 = tau1;
                                    f2 = tau2;
                                    let mut tail_ext = 0.0f64;
                                    if vb == NULLS {
                                        tail_ext = dispersion(pout_x, pout_y, pout_z, mu)
                                            * (tau2 - tau_total);
                                    }
                                    // add_phonon_higher_ratio
                                    let algo = f64::cast_from(2 * ord - 1) / f64::cast_from(ord);
                                    probability = algo * 2.0 * alpha * delta_t / PI
                                        * ((norm3(qx, qy, qz) * P0
                                            + dot3(qx, qy, qz, pmx, pmy, pmz))
                                            * (tau2 - tau1)
                                            - tail_ext)
                                            .exp()
                                        * (1.0
                                            - (-(phonon_lambda(qx, qy, qz) * (max_tau - tau1)))
                                                .exp());
                                }
                            }
                        }
                    }
                } else if upd == 3 {
                    // --- RemovePhonon ---
                    if ord == min_order {
                        probability = 0.0;
                    } else if ord == 1 {
                        v1 = hd;
                        v2 = tl;
                        qx = q[3 * (base + hd)];
                        qy = q[3 * (base + hd) + 1];
                        qz = q[3 * (base + hd) + 2];
                        // remove_phonon_zero_ratio
                        let e = 1.0
                            + (norm_sq3(qx, qy, qz) / 2.0
                                - dot3(qx, qy, qz, pout_x, pout_y, pout_z));
                        let t = 1.0 + norm3(qx, qy, qz) / P0;
                        probability = PI / (2.0 * alpha) * (e * tau_total).exp() / (t * t);
                    } else {
                        let r0 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                        k += 2;
                        let mut v = storage[base + uniform_index(r0, n_vertices) as usize] as usize;
                        while v == hd || link[base + v] as usize == hd {
                            let r = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                            k += 2;
                            v = storage[base + uniform_index(r, n_vertices) as usize] as usize;
                        }
                        let mut left = v;
                        let mut right = link[base + v] as usize;
                        if tau[base + left] > tau[base + right] {
                            let tmp = left;
                            left = right;
                            right = tmp;
                        }
                        v1 = left;
                        v2 = right;
                        qx = q[3 * (base + left)];
                        qy = q[3 * (base + left) + 1];
                        qz = q[3 * (base + left) + 2];

                        let mut blocked = false;
                        if ord != 2 {
                            let second_last = prev[base + tl] as usize;
                            let mut slot = next[base + left] as usize;
                            while slot != right {
                                if phonons_above[base + slot] == 1 && slot != second_last {
                                    blocked = true;
                                }
                                slot = next[base + slot] as usize;
                            }
                        }
                        if blocked {
                            probability = 0.0;
                        } else {
                            let delta_t = -tau[base + prev[base + left] as usize]
                                + if next[base + left] as usize == right {
                                    tau[base + next[base + right] as usize]
                                } else {
                                    tau[base + next[base + left] as usize]
                                };
                            // get_p_mean_range(left, right, q)
                            let mut pmx = 0.0f64;
                            let mut pmy = 0.0f64;
                            let mut pmz = 0.0f64;
                            let mut slot = left;
                            while slot != right {
                                let nx = next[base + slot] as usize;
                                let dt = tau[base + nx] - tau[base + slot];
                                pmx += (p_out[3 * (base + slot)] + qx) * dt;
                                pmy += (p_out[3 * (base + slot) + 1] + qy) * dt;
                                pmz += (p_out[3 * (base + slot) + 2] + qz) * dt;
                                slot = nx;
                            }
                            let inv = 1.0 / (tau[base + right] - tau[base + left]);
                            pmx *= inv;
                            pmy *= inv;
                            pmz *= inv;

                            let mut tail_ext = 0.0f64;
                            if next[base + right] == NULL {
                                tail_ext = dispersion(pout_x, pout_y, pout_z, mu)
                                    * (tau[base + right] - tau[base + prev[base + right] as usize]);
                            }
                            // remove_phonon_higher_ratio
                            let algo = f64::cast_from(ord - 1) / f64::cast_from(2 * ord - 3);
                            probability = algo * PI / (2.0 * alpha * delta_t)
                                * (-(norm3(qx, qy, qz) * P0 + dot3(qx, qy, qz, pmx, pmy, pmz))
                                    * (tau[base + right] - tau[base + left])
                                    + tail_ext)
                                    .exp()
                                / (1.0
                                    - (-(phonon_lambda(qx, qy, qz) * (max_tau - tau[base + left])))
                                        .exp());
                        }
                    }
                } else if upd == 4 {
                    // --- RescaleDiagram ---
                    if ord > 1 {
                        let mut energy = -mu;
                        // literal init (not the F64_MAX const path); see the literal-select note
                        let mut min_delta = 1.797_693_134_862_315_7e308f64;
                        let mut slot = hd;
                        while slot != tl {
                            let nx = next[base + slot] as usize;
                            let delta = tau[base + nx] - tau[base + slot];
                            if delta < min_delta {
                                min_delta = delta;
                            }
                            let delta_s = delta / tau_total;
                            let lk = link[base + slot];
                            let incoming = lk != NULL && tau[base + lk as usize] < tau[base + slot];
                            let count = if incoming {
                                phonons_above[base + slot]
                            } else {
                                phonons_above[base + slot] + 1
                            };
                            energy += delta_s
                                * (bare_dispersion(
                                    p_out[3 * (base + slot)],
                                    p_out[3 * (base + slot) + 1],
                                    p_out[3 * (base + slot) + 2],
                                ) + f64::cast_from(count));
                            slot = nx;
                        }
                        let n = f64::cast_from(ord - 1);
                        let sigma = (2.0 * n).sqrt() / energy;
                        if is_finite_f64(sigma) && sigma >= 0.0 {
                            let r1 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                            let r2 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k + 2);
                            k += 4;
                            f1 = normal_from_uniforms(r1, r2, 2.0 * n / energy, sigma);
                            // second condition: rescale collapse guard (see flat::updates)
                            if f1 >= 0.0
                                && f1 <= max_tau
                                && is_finite_f64(f1)
                                && min_delta * (f1 / tau_total) >= 10.0 * DELTA_TAU_LIMIT
                            {
                                // rescale_diagram_ratio
                                let acc = (2.0 * n * (f1 / tau_total).ln()
                                    - energy * (f1 - tau_total)
                                    + ((energy * f1 - 2.0 * n) * (energy * f1 - 2.0 * n)
                                        - (energy * tau_total - 2.0 * n)
                                            * (energy * tau_total - 2.0 * n))
                                        / (4.0 * n))
                                    .exp();
                                if is_finite_f64(acc) {
                                    probability = acc;
                                }
                            }
                        }
                    }
                } else if upd == 5 {
                    // --- ChangeQModulus ---
                    if ord > 0 {
                        let r0 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                        k += 2;
                        let va = storage[base + uniform_index(r0, n_vertices) as usize] as usize;
                        let vb = link[base + va] as usize;
                        let mut left = va;
                        let mut right = vb;
                        if tau[base + vb] < tau[base + va] {
                            left = vb;
                            right = va;
                        }
                        qx = q[3 * (base + left)];
                        qy = q[3 * (base + left) + 1];
                        qz = q[3 * (base + left) + 2];
                        let q_norm = norm3(qx, qy, qz);
                        if q_norm != 0.0 {
                            let mut pmx = 0.0f64;
                            let mut pmy = 0.0f64;
                            let mut pmz = 0.0f64;
                            let mut slot = left;
                            while slot != right {
                                let nx = next[base + slot] as usize;
                                let dt = tau[base + nx] - tau[base + slot];
                                pmx += (p_out[3 * (base + slot)] + qx) * dt;
                                pmy += (p_out[3 * (base + slot) + 1] + qy) * dt;
                                pmz += (p_out[3 * (base + slot) + 2] + qz) * dt;
                                slot = nx;
                            }
                            let inv = 1.0 / (tau[base + right] - tau[base + left]);
                            pmx *= inv;
                            pmy *= inv;
                            pmz *= inv;
                            let q0 = dot3(pmx, pmy, pmz, qx / q_norm, qy / q_norm, qz / q_norm);
                            // change_q_modulus_sigma, MASS = 1
                            let sigma = (1.0 / (tau[base + right] - tau[base + left])).sqrt();
                            if is_finite_f64(sigma) && sigma >= 0.0 {
                                let r1 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                                let r2 =
                                    u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k + 2);
                                k += 4;
                                f1 = normal_from_uniforms(r1, r2, q0, sigma);
                                v1 = left;
                                v2 = right;
                                if f1 >= 0.0 && is_finite_f64(f1) {
                                    probability = 1.0;
                                }
                            }
                        }
                    }
                } else if upd == 6 {
                    // --- ChangeQDirection ---
                    if ord > 0 {
                        let r0 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                        k += 2;
                        let va = storage[base + uniform_index(r0, n_vertices) as usize] as usize;
                        let vb = link[base + va] as usize;
                        let mut left = va;
                        let mut right = vb;
                        if tau[base + vb] < tau[base + va] {
                            left = vb;
                            right = va;
                        }
                        let ax = q[3 * (base + left)];
                        let ay = q[3 * (base + left) + 1];
                        let az = q[3 * (base + left) + 2];
                        let mut pmx = 0.0f64;
                        let mut pmy = 0.0f64;
                        let mut pmz = 0.0f64;
                        let mut slot = left;
                        while slot != right {
                            let nx = next[base + slot] as usize;
                            let dt = tau[base + nx] - tau[base + slot];
                            pmx += (p_out[3 * (base + slot)] + ax) * dt;
                            pmy += (p_out[3 * (base + slot) + 1] + ay) * dt;
                            pmz += (p_out[3 * (base + slot) + 2] + az) * dt;
                            slot = nx;
                        }
                        let inv = 1.0 / (tau[base + right] - tau[base + left]);
                        pmx *= inv;
                        pmy *= inv;
                        pmz *= inv;
                        let q_norm = norm3(ax, ay, az);
                        // change_q_direction_a, MASS = 1
                        let a =
                            (tau[base + right] - tau[base + left]) * norm3(pmx, pmy, pmz) * q_norm;
                        if a.abs() >= EPS && is_finite_f64(a) {
                            let r1 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                            let r2 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k + 2);
                            k += 4;
                            let phi = 2.0 * PI * r1;
                            let log_val = (-2.0 * r2 * a.sinh() + a.exp()).ln();
                            let theta = (log_val / a).acos();
                            let theta_base = (pmz / norm3(pmx, pmy, pmz)).acos();
                            let phi_base = pmy.atan2(pmx);
                            // spherical_to_cartesian then rotate_z_then_y
                            let sx = q_norm * phi.cos() * theta.sin();
                            let sy = q_norm * phi.sin() * theta.sin();
                            let sz = q_norm * theta.cos();
                            let st = theta_base.sin();
                            let ct = theta_base.cos();
                            let sp = phi_base.sin();
                            let cp = phi_base.cos();
                            let wx = ct * sx + st * sz;
                            let wy = sy;
                            let wz = -st * sx + ct * sz;
                            qx = cp * wx - sp * wy;
                            qy = sp * wx + cp * wy;
                            qz = wz;
                            v1 = left;
                            v2 = right;
                            // not-NaN check (NaN != NaN)
                            if qx == qx && qy == qy && qz == qz {
                                probability = 1.0;
                            }
                        }
                    }
                } else {
                    // --- ChangeTopology ---
                    if ord > 0 {
                        let r0 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                        k += 2;
                        let mut va = storage[base + uniform_index(r0, n_vertices) as usize] as usize;
                        if next[base + va] == NULL {
                            va = prev[base + va] as usize;
                        }
                        let vb = next[base + va] as usize;
                        let lk_a = link[base + va];
                        let lk_b = link[base + vb];
                        let out_a = lk_a != NULL && tau[base + lk_a as usize] > tau[base + va];
                        let out_b = lk_b != NULL && tau[base + lk_b as usize] > tau[base + vb];
                        let in_a = lk_a != NULL && tau[base + lk_a as usize] < tau[base + va];
                        let in_b = lk_b != NULL && tau[base + lk_b as usize] < tau[base + vb];
                        // literal-selects via statement-if; see change_internal_tau_lambda note
                        let mut c1 = 1.0f64;
                        if !out_a {
                            c1 = -c1;
                        }
                        let mut c2 = 1.0f64;
                        if !out_b {
                            c2 = -c2;
                        }
                        if lk_a as usize != vb {
                            qx = p_out[3 * (base + va)] + c1 * q[3 * (base + va)]
                                - c2 * q[3 * (base + vb)];
                            qy = p_out[3 * (base + va) + 1] + c1 * q[3 * (base + va) + 1]
                                - c2 * q[3 * (base + vb) + 1];
                            qz = p_out[3 * (base + va) + 2] + c1 * q[3 * (base + va) + 2]
                                - c2 * q[3 * (base + vb) + 2];
                            let special = out_a
                                && in_b
                                && (phonons_above[base + va] == 1
                                    || phonons_above[base + vb] == 1);
                            if !special {
                                v1 = va;
                                v2 = vb;
                                // change_topology_ratio, OMEGA = 1
                                let acc = (-(tau[base + vb] - tau[base + va])
                                    * (bare_dispersion(qx, qy, qz)
                                        - bare_dispersion(
                                            p_out[3 * (base + va)],
                                            p_out[3 * (base + va) + 1],
                                            p_out[3 * (base + va) + 2],
                                        )
                                        - (c1 - c2)))
                                    .exp();
                                if is_finite_f64(acc) {
                                    probability = acc;
                                }
                                // f1/f2 carry the phonons_above adjustment flags into accept
                                if in_a && out_b {
                                    f1 = 1.0;
                                }
                                if out_a && in_b {
                                    f2 = 1.0;
                                }
                            }
                        }
                    }
                }

                // --- Metropolis decision (mirrors step_selected; ratio() == 1) ---
                let impossible = probability < 0.0;
                if probability >= 1.0 {
                    accepted = true;
                } else if probability > 0.0 {
                    let r = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, k);
                    if r < probability {
                        accepted = true;
                    }
                }

                // --- Accept: apply the state mutation ---
                if accepted {
                    if upd == 0 || upd == 1 {
                        tau[base + v1] = f1;
                    } else if upd == 2 {
                        if ord == 0 {
                            // set_to_fake_order_one
                            q[3 * (base + hd)] = qx;
                            q[3 * (base + hd) + 1] = qy;
                            q[3 * (base + hd) + 2] = qz;
                            q[3 * (base + tl)] = qx;
                            q[3 * (base + tl) + 1] = qy;
                            q[3 * (base + tl) + 2] = qz;
                            p_out[3 * (base + hd)] -= qx;
                            p_out[3 * (base + hd) + 1] -= qy;
                            p_out[3 * (base + hd) + 2] -= qz;
                            link[base + hd] = tl as u32;
                            link[base + tl] = hd as u32;
                            order[chain] = 1;
                        } else {
                            // insert_arc_between(v1, v2, f1, f2, q); capacity was pre-checked
                            let left = v1;
                            let before_right = v2;

                            let lk_left = link[base + left];
                            let out_left =
                                lk_left != NULL && tau[base + lk_left as usize] > tau[base + left];
                            let mut phonons1 = phonons_above[base + left];
                            if out_left {
                                phonons1 += 1;
                            }
                            // splice_after(left, f1, p_out[left], q) -> new1
                            let mut new1 = 0usize;
                            while storage_idx[base + new1] != NULL {
                                new1 += 1;
                            }
                            let right1 = next[base + left];
                            tau[base + new1] = f1;
                            p_out[3 * (base + new1)] = p_out[3 * (base + left)];
                            p_out[3 * (base + new1) + 1] = p_out[3 * (base + left) + 1];
                            p_out[3 * (base + new1) + 2] = p_out[3 * (base + left) + 2];
                            q[3 * (base + new1)] = qx;
                            q[3 * (base + new1) + 1] = qy;
                            q[3 * (base + new1) + 2] = qz;
                            link[base + new1] = NULL;
                            phonons_above[base + new1] = 0;
                            prev[base + new1] = left as u32;
                            next[base + new1] = right1;
                            next[base + left] = new1 as u32;
                            if right1 == NULL {
                                tail[chain] = new1 as u32;
                            } else {
                                prev[base + right1 as usize] = new1 as u32;
                            }
                            let len1 = storage_len[chain] as usize;
                            storage[base + len1] = new1 as u32;
                            storage_idx[base + new1] = len1 as u32;
                            storage_len[chain] = (len1 + 1) as u32;
                            phonons_above[base + new1] = phonons1;

                            let right_left = if before_right == NULLS {
                                tail[chain] as usize
                            } else {
                                prev[base + before_right] as usize
                            };
                            let lk_rl = link[base + right_left];
                            let out_rl = lk_rl != NULL
                                && tau[base + lk_rl as usize] > tau[base + right_left];
                            let mut phonons2 = phonons_above[base + right_left];
                            if out_rl {
                                phonons2 += 1;
                            }
                            // splice_after(right_left, f2, p_out[right_left], q) -> new2
                            let mut new2 = 0usize;
                            while storage_idx[base + new2] != NULL {
                                new2 += 1;
                            }
                            let right2 = next[base + right_left];
                            tau[base + new2] = f2;
                            p_out[3 * (base + new2)] = p_out[3 * (base + right_left)];
                            p_out[3 * (base + new2) + 1] = p_out[3 * (base + right_left) + 1];
                            p_out[3 * (base + new2) + 2] = p_out[3 * (base + right_left) + 2];
                            q[3 * (base + new2)] = qx;
                            q[3 * (base + new2) + 1] = qy;
                            q[3 * (base + new2) + 2] = qz;
                            link[base + new2] = NULL;
                            phonons_above[base + new2] = 0;
                            prev[base + new2] = right_left as u32;
                            next[base + new2] = right2;
                            next[base + right_left] = new2 as u32;
                            if right2 == NULL {
                                tail[chain] = new2 as u32;
                            } else {
                                prev[base + right2 as usize] = new2 as u32;
                            }
                            let len2 = storage_len[chain] as usize;
                            storage[base + len2] = new2 as u32;
                            storage_idx[base + new2] = len2 as u32;
                            storage_len[chain] = (len2 + 1) as u32;
                            phonons_above[base + new2] = phonons2;

                            link[base + new1] = new2 as u32;
                            link[base + new2] = new1 as u32;

                            let mut slot = new1;
                            while slot != new2 {
                                p_out[3 * (base + slot)] -= qx;
                                p_out[3 * (base + slot) + 1] -= qy;
                                p_out[3 * (base + slot) + 2] -= qz;
                                slot = next[base + slot] as usize;
                            }
                            slot = next[base + new1] as usize;
                            while slot != new2 {
                                phonons_above[base + slot] += 1;
                                slot = next[base + slot] as usize;
                            }
                            order[chain] = ord + 1;
                        }
                    } else if upd == 3 {
                        if ord == 1 {
                            // clear_fake_order_one
                            p_out[3 * (base + hd)] += qx;
                            p_out[3 * (base + hd) + 1] += qy;
                            p_out[3 * (base + hd) + 2] += qz;
                            q[3 * (base + hd)] = 0.0;
                            q[3 * (base + hd) + 1] = 0.0;
                            q[3 * (base + hd) + 2] = 0.0;
                            q[3 * (base + tl)] = 0.0;
                            q[3 * (base + tl) + 1] = 0.0;
                            q[3 * (base + tl) + 2] = 0.0;
                            link[base + hd] = NULL;
                            link[base + tl] = NULL;
                            phonons_above[base + hd] = 0;
                            phonons_above[base + tl] = 0;
                            order[chain] = 0;
                        } else {
                            // remove_arc(v1, v2): v1 = left, v2 = right (ordered in attempt)
                            let left = v1;
                            let right = v2;
                            let mut slot = next[base + left] as usize;
                            while slot != right {
                                p_out[3 * (base + slot)] += qx;
                                p_out[3 * (base + slot) + 1] += qy;
                                p_out[3 * (base + slot) + 2] += qz;
                                phonons_above[base + slot] -= 1;
                                slot = next[base + slot] as usize;
                            }
                            // unlink(left), then unlink(right) — written out twice; the second
                            // block sees the links the first block rewired, exactly like the CPU.
                            let pv1 = prev[base + left];
                            let nx1 = next[base + left];
                            if pv1 == NULL {
                                head[chain] = nx1;
                            } else {
                                next[base + pv1 as usize] = nx1;
                            }
                            if nx1 == NULL {
                                tail[chain] = pv1;
                            } else {
                                prev[base + nx1 as usize] = pv1;
                            }
                            // swap_remove_storage(left)
                            let idx1 = storage_idx[base + left] as usize;
                            let last1 = storage_len[chain] as usize - 1;
                            storage[base + idx1] = storage[base + last1];
                            storage_len[chain] = last1 as u32;
                            storage_idx[base + left] = NULL;
                            if idx1 < last1 {
                                let moved = storage[base + idx1] as usize;
                                storage_idx[base + moved] = idx1 as u32;
                            }
                            // clear_vertex(left)
                            tau[base + left] = 0.0;
                            p_out[3 * (base + left)] = 0.0;
                            p_out[3 * (base + left) + 1] = 0.0;
                            p_out[3 * (base + left) + 2] = 0.0;
                            q[3 * (base + left)] = 0.0;
                            q[3 * (base + left) + 1] = 0.0;
                            q[3 * (base + left) + 2] = 0.0;
                            link[base + left] = NULL;
                            prev[base + left] = NULL;
                            next[base + left] = NULL;
                            phonons_above[base + left] = 0;

                            let pv2 = prev[base + right];
                            let nx2 = next[base + right];
                            if pv2 == NULL {
                                head[chain] = nx2;
                            } else {
                                next[base + pv2 as usize] = nx2;
                            }
                            if nx2 == NULL {
                                tail[chain] = pv2;
                            } else {
                                prev[base + nx2 as usize] = pv2;
                            }
                            // swap_remove_storage(right)
                            let idx2 = storage_idx[base + right] as usize;
                            let last2 = storage_len[chain] as usize - 1;
                            storage[base + idx2] = storage[base + last2];
                            storage_len[chain] = last2 as u32;
                            storage_idx[base + right] = NULL;
                            if idx2 < last2 {
                                let moved = storage[base + idx2] as usize;
                                storage_idx[base + moved] = idx2 as u32;
                            }
                            // clear_vertex(right)
                            tau[base + right] = 0.0;
                            p_out[3 * (base + right)] = 0.0;
                            p_out[3 * (base + right) + 1] = 0.0;
                            p_out[3 * (base + right) + 2] = 0.0;
                            q[3 * (base + right)] = 0.0;
                            q[3 * (base + right) + 1] = 0.0;
                            q[3 * (base + right) + 2] = 0.0;
                            link[base + right] = NULL;
                            prev[base + right] = NULL;
                            next[base + right] = NULL;
                            phonons_above[base + right] = 0;

                            order[chain] = ord - 1;
                        }
                    } else if upd == 4 {
                        // scale_taus(f1 / tau_total)
                        let scale = f1 / tau_total;
                        let mut slot = hd;
                        while slot != NULLS {
                            tau[base + slot] *= scale;
                            slot = next[base + slot] as usize;
                        }
                    } else if upd == 5 {
                        // update_arc_q(v1, v2, unit(q) * f1)
                        let q_norm = norm3(qx, qy, qz);
                        let nqx = qx / q_norm * f1;
                        let nqy = qy / q_norm * f1;
                        let nqz = qz / q_norm * f1;
                        let mut slot = v1;
                        while slot != v2 {
                            p_out[3 * (base + slot)] += qx;
                            p_out[3 * (base + slot)] -= nqx;
                            p_out[3 * (base + slot) + 1] += qy;
                            p_out[3 * (base + slot) + 1] -= nqy;
                            p_out[3 * (base + slot) + 2] += qz;
                            p_out[3 * (base + slot) + 2] -= nqz;
                            slot = next[base + slot] as usize;
                        }
                        q[3 * (base + v1)] = nqx;
                        q[3 * (base + v1) + 1] = nqy;
                        q[3 * (base + v1) + 2] = nqz;
                        q[3 * (base + v2)] = nqx;
                        q[3 * (base + v2) + 1] = nqy;
                        q[3 * (base + v2) + 2] = nqz;
                    } else if upd == 6 {
                        // update_arc_q(v1, v2, q_prime)
                        let ox = q[3 * (base + v1)];
                        let oy = q[3 * (base + v1) + 1];
                        let oz = q[3 * (base + v1) + 2];
                        let mut slot = v1;
                        while slot != v2 {
                            p_out[3 * (base + slot)] += ox;
                            p_out[3 * (base + slot)] -= qx;
                            p_out[3 * (base + slot) + 1] += oy;
                            p_out[3 * (base + slot) + 1] -= qy;
                            p_out[3 * (base + slot) + 2] += oz;
                            p_out[3 * (base + slot) + 2] -= qz;
                            slot = next[base + slot] as usize;
                        }
                        q[3 * (base + v1)] = qx;
                        q[3 * (base + v1) + 1] = qy;
                        q[3 * (base + v1) + 2] = qz;
                        q[3 * (base + v2)] = qx;
                        q[3 * (base + v2) + 1] = qy;
                        q[3 * (base + v2) + 2] = qz;
                    } else {
                        // ChangeTopology accept; f1/f2 carry (in1 && out2) / (out1 && in2)
                        if f1 == 1.0 {
                            phonons_above[base + v1] += 1;
                            phonons_above[base + v2] += 1;
                        }
                        if f2 == 1.0 && link[base + v1] as usize != v2 {
                            phonons_above[base + v1] -= 1;
                            phonons_above[base + v2] -= 1;
                        }
                        p_out[3 * (base + v1)] = qx;
                        p_out[3 * (base + v1) + 1] = qy;
                        p_out[3 * (base + v1) + 2] = qz;
                        if link[base + v1] as usize != v2 {
                            // swap_arc_connectivity(v1, v2)
                            let link_i = link[base + v1] as usize;
                            let link_j = link[base + v2] as usize;
                            let t0 = q[3 * (base + v1)];
                            let t1 = q[3 * (base + v1) + 1];
                            let t2 = q[3 * (base + v1) + 2];
                            q[3 * (base + v1)] = q[3 * (base + v2)];
                            q[3 * (base + v1) + 1] = q[3 * (base + v2) + 1];
                            q[3 * (base + v1) + 2] = q[3 * (base + v2) + 2];
                            q[3 * (base + v2)] = t0;
                            q[3 * (base + v2) + 1] = t1;
                            q[3 * (base + v2) + 2] = t2;
                            link[base + v1] = link_j as u32;
                            link[base + v2] = link_i as u32;
                            link[base + link_j] = v1 as u32;
                            link[base + link_i] = v2 as u32;
                        }
                    }
                }

                // --- Update statistics (chain-major, summed on host) ---
                let stat = chain * 8 + upd as usize;
                stats_proposed[stat] += 1;
                if impossible {
                    stats_impossible[stat] += 1;
                }
                if accepted {
                    stats_accepted[stat] += 1;
                }
            }

            // --- Per-cycle measurement sample ---
            let tl = tail[chain] as usize;
            let ord = order[chain];
            let tau_total = tau[base + tl];
            let mut exact = 0.0f64;
            if ord > 0 && tau_total >= 0.0 && tau_total <= max_tau {
                // t0 = bin_center(bin_index(tau_total)) on the [0, max_tau] grid
                let gstep = max_tau / f64::cast_from(num_bins);
                let mut bin = (tau_total / gstep) as u32;
                if bin >= num_bins {
                    bin = num_bins - 1;
                }
                let t0 = 0.5 * (gstep * f64::cast_from(bin) + gstep * f64::cast_from(bin + 1));

                // exact_estimator(t0)
                let lambda = t0 / tau_total - 1.0;
                let hd = head[chain] as usize;
                let mut electron_sum = 0.0f64;
                let mut slot = hd;
                while slot != tl {
                    let nx = next[base + slot] as usize;
                    electron_sum += dispersion(
                        p_out[3 * (base + slot)],
                        p_out[3 * (base + slot) + 1],
                        p_out[3 * (base + slot) + 2],
                        mu,
                    ) * (tau[base + nx] - tau[base + slot]);
                    slot = nx;
                }
                let mut phonon_sum = 0.0f64;
                slot = hd;
                while slot != NULLS {
                    let lk = link[base + slot] as usize;
                    let delta = tau[base + slot] - tau[base + lk];
                    if delta > 0.0 {
                        phonon_sum += delta;
                    }
                    slot = next[base + slot] as usize;
                }
                exact = (t0 / tau_total).powf(f64::cast_from(2 * (ord - 1)))
                    * (-(lambda * (electron_sum + phonon_sum))).exp();
            }
            let sample = cycle as usize * n_chains as usize + chain;
            samples_tau[sample] = tau_total;
            samples_exact[sample] = exact;
            samples_order[sample] = ord;
        }
    }
}

/// One kernel launch: `n_cycles` cycles of `steps_per_cycle` steps (the final cycle runs
/// `last_cycle_steps`), starting at global step counter `step0`.
#[derive(Clone, Copy, Debug)]
pub struct SegmentParams {
    pub step0: u64,
    pub n_cycles: u32,
    pub steps_per_cycle: u32,
    pub last_cycle_steps: u32,
}

/// Device handles for one resident population of chains.
pub struct DeviceState<R: Runtime> {
    pub client: ComputeClient<R>,
    pub chains: u32,
    pub capacity: u32,
    pub tau: cubecl::server::Handle,
    pub p_out: cubecl::server::Handle,
    pub q: cubecl::server::Handle,
    pub link: cubecl::server::Handle,
    pub prev: cubecl::server::Handle,
    pub next: cubecl::server::Handle,
    pub storage_idx: cubecl::server::Handle,
    pub phonons_above: cubecl::server::Handle,
    pub storage: cubecl::server::Handle,
    pub storage_len: cubecl::server::Handle,
    pub head: cubecl::server::Handle,
    pub tail: cubecl::server::Handle,
    pub order: cubecl::server::Handle,
}

impl<R: Runtime> DeviceState<R> {
    pub fn upload(client: ComputeClient<R>, buffers: &crate::gpu::state::GpuStateBuffers) -> Self {
        let flat3 =
            |v: &Vec<[f64; 3]>| -> Vec<f64> { v.iter().flat_map(|p| p.iter().copied()).collect() };
        Self {
            chains: buffers.chains as u32,
            capacity: buffers.capacity as u32,
            tau: client.create_from_slice(f64::as_bytes(&buffers.tau)),
            p_out: client.create_from_slice(f64::as_bytes(&flat3(&buffers.p_out))),
            q: client.create_from_slice(f64::as_bytes(&flat3(&buffers.q))),
            link: client.create_from_slice(u32::as_bytes(&buffers.link)),
            prev: client.create_from_slice(u32::as_bytes(&buffers.prev)),
            next: client.create_from_slice(u32::as_bytes(&buffers.next)),
            storage_idx: client.create_from_slice(u32::as_bytes(&buffers.storage_idx)),
            phonons_above: client.create_from_slice(u32::as_bytes(&buffers.phonons_above)),
            storage: client.create_from_slice(u32::as_bytes(&buffers.storage)),
            storage_len: client.create_from_slice(u32::as_bytes(&buffers.storage_len)),
            head: client.create_from_slice(u32::as_bytes(&buffers.head)),
            tail: client.create_from_slice(u32::as_bytes(&buffers.tail)),
            order: client.create_from_slice(u32::as_bytes(&buffers.order)),
            client,
        }
    }

    /// Read the full chain state back into host buffers (final state, parity tests).
    pub fn download(&self) -> crate::gpu::state::GpuStateBuffers {
        let read_f64 = |h: &cubecl::server::Handle| {
            f64::from_bytes(&self.client.read_one_unchecked(h.clone())).to_vec()
        };
        let read_u32 = |h: &cubecl::server::Handle| {
            u32::from_bytes(&self.client.read_one_unchecked(h.clone())).to_vec()
        };
        let unflat3 =
            |v: Vec<f64>| -> Vec<[f64; 3]> { v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect() };
        crate::gpu::state::GpuStateBuffers {
            chains: self.chains as usize,
            capacity: self.capacity as usize,
            tau: read_f64(&self.tau),
            p_out: unflat3(read_f64(&self.p_out)),
            q: unflat3(read_f64(&self.q)),
            link: read_u32(&self.link),
            prev: read_u32(&self.prev),
            next: read_u32(&self.next),
            storage_idx: read_u32(&self.storage_idx),
            phonons_above: read_u32(&self.phonons_above),
            storage: read_u32(&self.storage),
            storage_len: read_u32(&self.storage_len),
            head: read_u32(&self.head),
            tail: read_u32(&self.tail),
            order: read_u32(&self.order),
        }
    }
}

/// Per-segment host-side output: one `(tau, order, exact)` sample per chain per cycle
/// (cycle-major) plus per-update-type counters summed over chains.
pub struct SegmentOutput {
    pub samples_tau: Vec<f64>,
    pub samples_exact: Vec<f64>,
    pub samples_order: Vec<u32>,
    pub proposed: [u64; 8],
    pub accepted: [u64; 8],
    pub impossible: [u64; 8],
}

/// Launch one segment and read back its sample log. Blocks until the device finishes.
pub fn launch_segment<R: Runtime>(
    state: &DeviceState<R>,
    cfg: &RunConfig,
    params: SegmentParams,
) -> SegmentOutput {
    assert_eq!(
        (crate::physics::MASS, crate::physics::OMEGA),
        (1.0, 1.0),
        "the GPU kernel folds MASS = OMEGA = 1"
    );
    assert!(params.n_cycles > 0);
    assert!(params.last_cycle_steps <= params.steps_per_cycle);

    let client = &state.client;
    let chains = state.chains;
    let n_samples = (params.n_cycles as usize) * (chains as usize);
    let samples_tau = client.empty(n_samples * core::mem::size_of::<f64>());
    let samples_exact = client.empty(n_samples * core::mem::size_of::<f64>());
    let samples_order = client.empty(n_samples * core::mem::size_of::<u32>());
    let zeros = vec![0u32; chains as usize * 8];
    let stats_proposed = client.create_from_slice(u32::as_bytes(&zeros));
    let stats_accepted = client.create_from_slice(u32::as_bytes(&zeros));
    let stats_impossible = client.create_from_slice(u32::as_bytes(&zeros));

    let group = DEFAULT_WORKGROUP_SIZE;
    let cube_count = chains.div_ceil(group);
    let state_len = (chains * state.capacity) as usize;
    let arr = |h: &cubecl::server::Handle, len: usize| -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(h.clone(), len) }
    };

    mc_kernel::launch::<R>(
        client,
        CubeCount::Static(cube_count, 1, 1),
        CubeDim::new_1d(group),
        arr(&state.tau, state_len),
        arr(&state.p_out, 3 * state_len),
        arr(&state.q, 3 * state_len),
        arr(&state.link, state_len),
        arr(&state.prev, state_len),
        arr(&state.next, state_len),
        arr(&state.storage_idx, state_len),
        arr(&state.phonons_above, state_len),
        arr(&state.storage, state_len),
        arr(&state.storage_len, chains as usize),
        arr(&state.head, chains as usize),
        arr(&state.tail, chains as usize),
        arr(&state.order, chains as usize),
        arr(&samples_tau, n_samples),
        arr(&samples_exact, n_samples),
        arr(&samples_order, n_samples),
        arr(&stats_proposed, chains as usize * 8),
        arr(&stats_accepted, chains as usize * 8),
        arr(&stats_impossible, chains as usize * 8),
        cfg.seed as u32,
        (cfg.seed >> 32) as u32,
        params.step0,
        chains,
        state.capacity,
        params.n_cycles,
        params.steps_per_cycle,
        params.last_cycle_steps,
        cfg.alpha,
        cfg.mu,
        cfg.max_tau,
        cfg.min_order as u32,
        cfg.max_order as u32,
        cfg.num_bins as u32,
    );

    let tau = f64::from_bytes(&client.read_one_unchecked(samples_tau)).to_vec();
    let exact = f64::from_bytes(&client.read_one_unchecked(samples_exact)).to_vec();
    let ord = u32::from_bytes(&client.read_one_unchecked(samples_order)).to_vec();
    let sum8 = |bytes: &[u8]| -> [u64; 8] {
        let counts = u32::from_bytes(bytes);
        let mut out = [0u64; 8];
        for (i, &c) in counts.iter().enumerate() {
            out[i % 8] += u64::from(c);
        }
        out
    };
    let proposed = sum8(&client.read_one_unchecked(stats_proposed));
    let accepted = sum8(&client.read_one_unchecked(stats_accepted));
    let impossible = sum8(&client.read_one_unchecked(stats_impossible));

    SegmentOutput {
        samples_tau: tau,
        samples_exact: exact,
        samples_order: ord,
        proposed,
        accepted,
        impossible,
    }
}
