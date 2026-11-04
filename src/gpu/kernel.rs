//! Phase A4: the CubeCL MC kernel — one ab-initio chain per cube.
//!
//! Architecture (dictated by the A1 result: only warp-cooperative strip loads
//! coalesce): **one chain per cube of `CUBE_DIM_X` lanes**. Every lane executes
//! the chain's update logic redundantly — Philox is counter-based, so all
//! lanes compute identical draws with no communication, and control flow is
//! lane-uniform by construction (no divergence, every `sync_cube` is reached
//! by the whole cube). Lanes cooperate only inside vertex evaluations:
//! [`arc_factor`] strides the rank dimension across lanes with a
//! shared-memory reduce, fusing the two dots of one arc factor over a shared
//! vtab strip (the dual-k optimization). State mutations happen on lane 0
//! between barriers.
//!
//! Lane-uniformity is a *hard* requirement, not a tuning choice. A cube is
//! CubeCL's thread block: lanes co-resident on one SM sharing one shared-
//! memory arena and `sync_cube` (= `__syncthreads`), a block-wide rendezvous
//! whose arrivals the hardware pairs by *dynamic count*, not source
//! location. If lanes diverged, either (a) barrier counts mismatch — some
//! lanes wait at a barrier the others never reach — and the kernel
//! deadlocks (SM reads busy, no progress), or (b) counts happen to match at
//! *different* call sites, the barrier "succeeds", and lanes in unrelated
//! phases race each other through the one shared `sm` arena — silent
//! bit-level corruption. The barrier count per step depends on the update
//! type, the proposal draws, and the diagram state (how many [`arc_factor`]
//! calls a step makes), so uniformity must hold over the whole dynamic
//! trace, not merely the update label — which the redundant-identical-lanes
//! design guarantees bit-for-bit. The only divergence, the strided rank
//! loop's `g < r` guard, contains no barrier and reconverges before the
//! reduce. (Why lockstep selection is statistically free is argued in
//! `flat::batched`.)
//!
//! CPU-runtime parity runs at `CubeDim = 1` (lane 0 only): `sync_cube`
//! no-ops, the reduce degenerates to the sequential sum in flat order, and
//! the CPU runtime's single shared-memory arena sees no cross-cube races
//! (its count-based barriers cannot align the *data-dependent* barrier
//! sequences this kernel produces at ≥ 2 lanes — CUDA blocks are independent
//! and unaffected). Production launches use `CubeDim = 32`.
//!
//! Everything mirrors `flat::updates` + `flat::weight` line by line: RNG word
//! order, guards, and floating-point association (`((bra*)·w)·v`, summed in g
//! order) — see the parity suite. Cube-dialect rules as in
//! `rmc-frohlich/src/gpu/kernel.rs` (statement-form ifs for literal selects,
//! literal inits for mutable vars, usize indexing, no tuple returns).

use cubecl::prelude::*;

/// `flat::NONE` in the u32 state arrays.
const NONE: u32 = 0xffff_ffff;
/// 2^-53, the scale of `rand`'s 53-bit `f64` conversion.
const U01_SCALE: f64 = 1.110_223_024_625_156_5e-16;
/// `f64::EPSILON`.
const EPS: f64 = 2.220_446_049_250_313e-16;
/// `rmc_diagmc::updates::DELTA_TAU_LIMIT` = 10 · f64::EPSILON.
const DELTA_TAU_LIMIT: f64 = 2.220_446_049_250_313e-15;
/// `reconstruct::RY_TO_EV` (asserted equal at launch).
pub const RY2EV: f64 = 13.605_698_066;
/// Shared-memory layout: 4 partials per lane (≤ 32 lanes), then scratch
/// slots for multi-value helper returns.
const SM_PARTIALS: u32 = 128;
const SM_LEN: u32 = 136;

// ---------------------------------------------------------------------------
// Philox4x32-10 counter stream (verbatim from rmc-frohlich/src/gpu/kernel.rs,
// which mirrors flat::philox exactly)
// ---------------------------------------------------------------------------

#[cube]
fn mul_hi(a: u32, b: u32) -> u32 {
    ((u64::cast_from(a) * u64::cast_from(b)) >> 32) as u32
}

#[cube]
fn mul_lo(a: u32, b: u32) -> u32 {
    (u64::cast_from(a) * u64::cast_from(b)) as u32
}

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

/// Word `word` of the CPU `PhiloxRng { seed, chain_id: id, step }` stream.
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

/// `rand`'s `gen::<f64>()`: words `word`, `word + 1` low-first, `(v >> 11)·2⁻⁵³`.
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

#[cube]
fn is_finite_f64(x: f64) -> bool {
    x.abs() <= 1.797_693_134_862_315_7e308
}

/// `exp(x) − 1` with a short series below |x| < 1e-4 (no expm1 in the dialect).
#[cube]
fn exp_m1(x: f64) -> f64 {
    if x.abs() < 1.0e-4 { x * (1.0 + x * (0.5 + x * (1.0 / 6.0))) } else { x.exp() - 1.0 }
}

#[cube]
fn exponential_sample_bounded(r: f64, lambda: f64, a: f64, b: f64) -> f64 {
    a - (r * exp_m1(-(lambda * (b - a)))).log1p() / lambda
}

#[cube]
fn exponential_pdf_bounded(x: f64, lambda: f64, a: f64, b: f64) -> f64 {
    lambda / -exp_m1(-(lambda * (b - a))) * (-(lambda * (x - a))).exp()
}

/// `uniform_index_from_u01`.
#[cube]
fn uniform_index(r: f64, len: u32) -> u32 {
    let idx = (r * f64::cast_from(len)) as u32;
    if idx >= len { len - 1 } else { idx }
}

// ---------------------------------------------------------------------------
// Grid index arithmetic (mirror flat::wrap_add / neg)
// ---------------------------------------------------------------------------

#[cube]
fn wrap_add(n: u32, a: u32, b: u32) -> u32 {
    let nn = n * n;
    let i = (a / nn + b / nn) % n;
    let j = (a / n % n + b / n % n) % n;
    let l = (a % n + b % n) % n;
    (i * n + j) * n + l
}

#[cube]
fn neg_idx(n: u32, a: u32) -> u32 {
    let nn = n * n;
    let i = (n - a / nn % n) % n;
    let j = (n - a / n % n) % n;
    let l = (n - a % n) % n;
    (i * n + j) * n + l
}

#[cube]
fn wrap_sub(n: u32, a: u32, b: u32) -> u32 {
    wrap_add(n, a, neg_idx(n, b))
}

// ---------------------------------------------------------------------------
// Positional searches on the live τ prefix (mirror flat partition_points)
// ---------------------------------------------------------------------------

/// First index in `0..n` with `tau[base + idx] >= x` (`partition_point(t < x)`).
#[cube]
fn partition_lt(tau: &Array<f64>, base: usize, n: usize, x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if tau[base + mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// First index in `0..n` with `tau[base + idx] > x` (`partition_point(t <= x)`).
#[cube]
fn partition_le(tau: &Array<f64>, base: usize, n: usize, x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if tau[base + mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `flat::FlatDiagram::segment_at`: last vertex with τ_v ≤ τ, clamped to tail−1.
#[cube]
fn segment_at(tau: &Array<f64>, base: usize, ln: usize, x: f64) -> usize {
    let p = partition_le(tau, base, ln, x);
    let mut key = 0usize;
    if p > 0 {
        key = p - 1;
    }
    if key > ln - 2 {
        key = ln - 2;
    }
    key
}

/// The proposal CDF search (`AbInitioModel::q_from_target`): first index with
/// `q_cdf[idx] > target`, clamped to nk−1.
#[cube]
fn cdf_search(q_cdf: &Array<f64>, nk: usize, target: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = nk;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if q_cdf[mid] <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo > nk - 1 {
        lo = nk - 1;
    }
    lo
}

// ---------------------------------------------------------------------------
// The cooperative arc factor F = g(k_emit,q,ν) · g(k_absorb,q,ν)*
// ---------------------------------------------------------------------------

/// Fused dual-k evaluation: lanes stride the rank dimension, loading the
/// shared vtab strip once, accumulating both dots' partials; every lane then
/// reduces the shared partials redundantly (lane order = flat order at
/// 1 lane). Association mirrors `short_indices_capped`: `((bra*)·w)·v`,
/// summed in g order, `· RY2EV` after the sum, `+ overlap·long` after that.
/// Returns F_re; F_im is left at `sm[SM_PARTIALS]` (read it before the next
/// helper call).
#[cube]
#[allow(clippy::too_many_arguments)]
fn arc_factor(
    w_re: &Array<f32>,
    w_im: &Array<f32>,
    v_re: &Array<f32>,
    v_im: &Array<f32>,
    cbm: &Array<f64>,
    bra_band: &Array<u32>,
    long_re: &Array<f64>,
    long_im: &Array<f64>,
    sm: &mut SharedMemory<f64>,
    k_emit: u32,
    k_absorb: u32,
    q: u32,
    nu: u32,
    grid_n: u32,
    rank: u32,
    nmodes: u32,
    nbnd: u32,
    lane: usize,
    lanes: usize,
) -> f64 {
    let kp_e = wrap_add(grid_n, k_emit, q);
    let kp_a = wrap_add(grid_n, k_absorb, q);
    let u_e = (k_emit * rank) as usize;
    let u_a = (k_absorb * rank) as usize;
    let v0 = ((q * nmodes + nu) * rank) as usize;
    let c_e = (kp_e * nbnd) as usize * 2;
    let c_a = (kp_a * nbnd) as usize * 2;

    let mut acc_e_re = 0.0f64;
    let mut acc_e_im = 0.0f64;
    let mut acc_a_re = 0.0f64;
    let mut acc_a_im = 0.0f64;
    let r = rank as usize;
    let iters = (r + lanes - 1) / lanes;
    for t in 0..iters {
        let g = lane + t * lanes;
        if g < r {
            let vr = f64::cast_from(v_re[v0 + g]);
            let vi = f64::cast_from(v_im[v0 + g]);
            let b = bra_band[g] as usize;

            let br = cbm[c_e + 2 * b];
            let bi = -cbm[c_e + 2 * b + 1];
            let wr = f64::cast_from(w_re[u_e + g]);
            let wi = f64::cast_from(w_im[u_e + g]);
            let t_re = br * wr - bi * wi;
            let t_im = br * wi + bi * wr;
            acc_e_re += t_re * vr - t_im * vi;
            acc_e_im += t_re * vi + t_im * vr;

            let br = cbm[c_a + 2 * b];
            let bi = -cbm[c_a + 2 * b + 1];
            let wr = f64::cast_from(w_re[u_a + g]);
            let wi = f64::cast_from(w_im[u_a + g]);
            let t_re = br * wr - bi * wi;
            let t_im = br * wi + bi * wr;
            acc_a_re += t_re * vr - t_im * vi;
            acc_a_im += t_re * vi + t_im * vr;
        }
    }
    sm[4 * lane] = acc_e_re;
    sm[4 * lane + 1] = acc_e_im;
    sm[4 * lane + 2] = acc_a_re;
    sm[4 * lane + 3] = acc_a_im;
    sync_cube();
    let mut s_e_re = 0.0f64;
    let mut s_e_im = 0.0f64;
    let mut s_a_re = 0.0f64;
    let mut s_a_im = 0.0f64;
    for l in 0..lanes {
        s_e_re += sm[4 * l];
        s_e_im += sm[4 * l + 1];
        s_a_re += sm[4 * l + 2];
        s_a_im += sm[4 * l + 3];
    }

    // Overlap ⟨kp|k⟩ = Σ_b cbm[kp,b]* · cbm[k,b], summed in b order.
    let mut ov_e_re = 0.0f64;
    let mut ov_e_im = 0.0f64;
    let mut ov_a_re = 0.0f64;
    let mut ov_a_im = 0.0f64;
    let ck_e = (k_emit * nbnd) as usize * 2;
    let ck_a = (k_absorb * nbnd) as usize * 2;
    for b in 0..nbnd as usize {
        let br = cbm[c_e + 2 * b];
        let bi = -cbm[c_e + 2 * b + 1];
        let kr = cbm[ck_e + 2 * b];
        let ki = cbm[ck_e + 2 * b + 1];
        ov_e_re += br * kr - bi * ki;
        ov_e_im += br * ki + bi * kr;
        let br = cbm[c_a + 2 * b];
        let bi = -cbm[c_a + 2 * b + 1];
        let kr = cbm[ck_a + 2 * b];
        let ki = cbm[ck_a + 2 * b + 1];
        ov_a_re += br * kr - bi * ki;
        ov_a_im += br * ki + bi * kr;
    }
    let l_idx = (q * nmodes + nu) as usize;
    let lr = long_re[l_idx];
    let li = long_im[l_idx];
    let g_e_re = s_e_re * RY2EV + (ov_e_re * lr - ov_e_im * li);
    let g_e_im = s_e_im * RY2EV + (ov_e_re * li + ov_e_im * lr);
    let g_a_re = s_a_re * RY2EV + (ov_a_re * lr - ov_a_im * li);
    let g_a_im = s_a_im * RY2EV + (ov_a_re * li + ov_a_im * lr);

    // F = g_e · g_a* .
    let f_re = g_e_re * g_a_re + g_e_im * g_a_im;
    let f_im = g_e_im * g_a_re - g_e_re * g_a_im;
    if lane == 0 {
        sm[SM_PARTIALS as usize] = f_im;
    }
    sync_cube();
    f_re
}

// ---------------------------------------------------------------------------
// Weight-ratio pieces (mirror flat::weight)
// ---------------------------------------------------------------------------

/// `flat::weight::segment_change`: Σ over segments overlapping (τ1, τ2) of
/// −(ε(p − q_ins) − ε(p))·overlap. Read-only, executed redundantly.
#[cube]
#[allow(clippy::too_many_arguments)]
fn segment_change(
    tau: &Array<f64>,
    p_out: &Array<u32>,
    electron_ev: &Array<f64>,
    base: usize,
    ln: usize,
    tau1: f64,
    tau2: f64,
    q_ins: u32,
    grid_n: u32,
) -> f64 {
    let mut key = segment_at(tau, base, ln, tau1);
    let mut out = 0.0f64;
    while key < ln - 1 && tau[base + key] < tau2 {
        let mut left = tau1;
        if tau[base + key] > left {
            left = tau[base + key];
        }
        let mut right = tau2;
        if tau[base + key + 1] < right {
            right = tau[base + key + 1];
        }
        if right > left {
            let p = p_out[base + key];
            let moved = wrap_sub(grid_n, p, q_ins);
            out -= (electron_ev[moved as usize] - electron_ev[p as usize]) * (right - left);
        }
        key += 1;
    }
    out
}

/// `flat::weight::changed_arc_factors`: log-modulus via return; the phase
/// (re, im) is left at `sm[SM_PARTIALS + 1]`, `sm[SM_PARTIALS + 2]` — read
/// both before the next helper call. `skip_left == NONE` means no skip.
#[cube]
#[allow(clippy::too_many_arguments)]
fn changed_arc_factors(
    tau: &Array<f64>,
    p_out: &Array<u32>,
    qv: &Array<u32>,
    branch: &Array<u32>,
    link: &Array<u32>,
    w_re: &Array<f32>,
    w_im: &Array<f32>,
    v_re: &Array<f32>,
    v_im: &Array<f32>,
    cbm: &Array<f64>,
    bra_band: &Array<u32>,
    long_re: &Array<f64>,
    long_im: &Array<f64>,
    sm: &mut SharedMemory<f64>,
    base: usize,
    ln: usize,
    tau1: f64,
    tau2: f64,
    q_ins: u32,
    skip_left: u32,
    skip_right: u32,
    grid_n: u32,
    rank: u32,
    nmodes: u32,
    nbnd: u32,
    lane: usize,
    lanes: usize,
) -> f64 {
    let mut log_modulus = 0.0f64;
    let mut phase_re = 1.0f64;
    let mut phase_im = 0.0f64;
    let start = partition_lt(tau, base, ln, tau1);
    let mut end = partition_le(tau, base, ln, tau2);
    if end > ln - 1 {
        end = ln - 1;
    }
    // Zero-modulus short-circuit mirrors the flat early return (skipping the
    // remaining arcs is load-bearing: ±inf from later arcs would give NaN).
    let mut dead = false;
    for key in start..end {
        let linked = link[base + key];
        // Body-level skip instead of `continue`: uniform across lanes, so the
        // arc_factor barriers stay aligned.
        if !dead && linked != NONE {
            let linked = linked as usize;
            let mut left = key;
            let mut right = linked;
            if linked < key {
                left = linked;
                right = key;
            }
            let old_emit = p_out[base + left];
            let old_absorb = p_out[base + right - 1];
            let emit_changed = tau[base + left] >= tau1 && tau[base + left] < tau2;
            let absorb_changed = tau[base + right] > tau1 && tau[base + right] <= tau2;
            let skipped =
                skip_left != NONE && left == skip_left as usize && right == skip_right as usize;
            let handled_at_left = key == right && emit_changed;
            if !skipped && (emit_changed || absorb_changed) && !handled_at_left {
                let q = qv[base + left];
                let nu = branch[base + left];
                let old_re = arc_factor(
                    w_re, w_im, v_re, v_im, cbm, bra_band, long_re, long_im, sm, old_emit,
                    old_absorb, q, nu, grid_n, rank, nmodes, nbnd, lane, lanes,
                );
                let old_im = sm[SM_PARTIALS as usize];
                let mut new_emit = old_emit;
                if emit_changed {
                    new_emit = wrap_sub(grid_n, old_emit, q_ins);
                }
                let mut new_absorb = old_absorb;
                if absorb_changed {
                    new_absorb = wrap_sub(grid_n, old_absorb, q_ins);
                }
                let new_re = arc_factor(
                    w_re, w_im, v_re, v_im, cbm, bra_band, long_re, long_im, sm, new_emit,
                    new_absorb, q, nu, grid_n, rank, nmodes, nbnd, lane, lanes,
                );
                let new_im = sm[SM_PARTIALS as usize];
                let old_modulus = (old_re * old_re + old_im * old_im).sqrt();
                let new_modulus = (new_re * new_re + new_im * new_im).sqrt();
                if old_modulus == 0.0 || new_modulus == 0.0 {
                    // ±inf via runtime division (a literal 1.0/0.0 trips the
                    // macro's const conversion).
                    let mut fzero = 0.0f64;
                    fzero *= 1.0;
                    log_modulus = 1.0 / fzero;
                    if new_modulus == 0.0 {
                        log_modulus = 0.0 - log_modulus;
                    }
                    phase_re = 1.0;
                    phase_im = 0.0;
                    dead = true;
                } else {
                    log_modulus += new_modulus.ln() - old_modulus.ln();
                    // phase *= (new/|new|) · (old/|old|)* .
                    let nr = new_re / new_modulus;
                    let ni = new_im / new_modulus;
                    let or_ = old_re / old_modulus;
                    let oi = -(old_im / old_modulus);
                    let t_re = nr * or_ - ni * oi;
                    let t_im = nr * oi + ni * or_;
                    let p_re = phase_re * t_re - phase_im * t_im;
                    let p_im = phase_re * t_im + phase_im * t_re;
                    phase_re = p_re;
                    phase_im = p_im;
                }
            }
        }
    }
    if lane == 0 {
        sm[SM_PARTIALS as usize + 1] = phase_re;
        sm[SM_PARTIALS as usize + 2] = phase_im;
    }
    sync_cube();
    log_modulus
}

// ---------------------------------------------------------------------------
// The MC kernel
// ---------------------------------------------------------------------------

/// Steps every chain `n_cycles` cycles forward, recording one
/// `(τ, E_loc, phase, order)` sample per cycle. State layout: slot `s` of
/// chain `c` at `c·capacity + s` (chain-major). Samples are cycle-major.
/// Update stats use the unified indices 0 = ChangeTau, 1 = ChangeInternalTau,
/// 2 = AddPhonon, 3 = RemovePhonon (a 3-update `fixed_tau` selection maps
/// onto 1..4).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn mc_kernel(
    tau: &mut Array<f64>,
    p_out: &mut Array<u32>,
    qv: &mut Array<u32>,
    branch: &mut Array<u32>,
    link: &mut Array<u32>,
    len: &mut Array<u32>,
    log_mod: &mut Array<f64>,
    phase_re: &mut Array<f64>,
    phase_im: &mut Array<f64>,
    w_re: &Array<f32>,
    w_im: &Array<f32>,
    v_re: &Array<f32>,
    v_im: &Array<f32>,
    cbm: &Array<f64>,
    bra_band: &Array<u32>,
    long_re: &Array<f64>,
    long_im: &Array<f64>,
    electron_ev: &Array<f64>,
    phonon_ev: &Array<f64>,
    q_cdf: &Array<f64>,
    q_weight: &Array<f64>,
    samples_tau: &mut Array<f64>,
    samples_eloc: &mut Array<f64>,
    samples_pre: &mut Array<f64>,
    samples_pim: &mut Array<f64>,
    samples_order: &mut Array<u32>,
    stats_proposed: &mut Array<u32>,
    stats_accepted: &mut Array<u32>,
    stats_impossible: &mut Array<u32>,
    seed_lo: u32,
    seed_hi: u32,
    step0: u64,
    chain_offset: u64,
    n_chains: u32,
    capacity: u32,
    n_cycles: u32,
    steps_per_cycle: u32,
    grid_n: u32,
    rank: u32,
    nmodes: u32,
    nbnd: u32,
    mu: f64,
    max_tau: f64,
    q_weight_sum: f64,
    ln_nq: f64,
    min_order: u32,
    max_order: u32,
    n_updates: u32,
) {
    let chain = CUBE_POS_X;
    if chain < n_chains {
        let lane = UNIT_POS_X as usize;
        let lanes = CUBE_DIM_X as usize;
        let base = (chain * capacity) as usize;
        let nk = (grid_n * grid_n * grid_n) as usize;
        let gid = chain_offset + u64::cast_from(chain);
        let id_lo = gid as u32;
        let id_hi = (gid >> 32) as u32;
        let mut sm = SharedMemory::<f64>::new(SM_LEN as usize);

        for cycle in 0..n_cycles {
            for s in 0..steps_per_cycle {
                let step = step0
                    + u64::cast_from(cycle) * u64::cast_from(steps_per_cycle)
                    + u64::cast_from(s);
                let step_lo = step as u32;
                let step_hi = (step >> 32) as u32;

                // Update-type selection keyed by (seed, chain, step)
                let sel = philox_block_word(
                    seed_lo,
                    seed_hi,
                    id_lo,
                    id_hi,
                    step_lo,
                    step_hi ^ 0xffff_ffffu32,
                    0u32,
                ) % n_updates;
                let mut upd = sel;
                if n_updates == 3 {
                    upd = sel + 1;
                }

                let ln = len[chain as usize] as usize;
                let tail = ln - 1;
                let ord = (ln - 2) / 2;
                let tau_ext = tau[base + tail];

                // Proposal registers shared across the update branches
                // (literal inits: the macro must make them runtime vars;
                // -1.0 is a Neg expr to the macro, so negate after init).
                let mut probability = 1.0f64;
                probability = -probability;
                let mut ratio_log = 0.0f64;
                let mut ratio_pre = 1.0f64;
                let mut ratio_pim = 0.0f64;
                let mut f1 = 0.0f64; // τ' | τ1
                let mut f2 = 0.0f64; // τ2
                let mut pq = 0u32;
                let mut pnu = 0u32;
                let mut a_left = 0usize;
                let mut a_right = 0usize;

                if upd == 0 {
                    // --- ChangeTau ---
                    let lower = tau[base + tail - 1];
                    if lower < max_tau {
                        let r = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, 0u32);
                        f1 = lower + r * (max_tau - lower);
                        if f1 - lower >= DELTA_TAU_LIMIT {
                            ratio_log = -(electron_ev[p_out[base + tail - 1] as usize] - mu)
                                * (f1 - tau_ext);
                            probability = ratio_log.exp();
                        }
                    }
                }

                if upd == 1 {
                    // --- ChangeInternalTau ---
                    if ord > 0 {
                        let r1 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, 0u32);
                        let key = 1 + uniform_index(r1, 2 * ord as u32) as usize;
                        let lower = tau[base + key - 1];
                        let upper = tau[base + key + 1];
                        let r2 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, 2u32);
                        f1 = lower + r2 * (upper - lower);
                        if f1 - lower >= DELTA_TAU_LIMIT && upper - f1 >= DELTA_TAU_LIMIT {
                            let delta = f1 - tau[base + key];
                            let before = electron_ev[p_out[base + key - 1] as usize];
                            let after = electron_ev[p_out[base + key] as usize];
                            let omega =
                                phonon_ev[(qv[base + key] * nmodes + branch[base + key]) as usize];
                            let mut phonon = -(omega * delta);
                            if link[base + key] as usize > key {
                                phonon = omega * delta;
                            }
                            ratio_log = (after - before) * delta + phonon;
                            probability = ratio_log.exp();
                            a_left = key;
                        }
                    }
                }

                if upd == 2 {
                    // --- AddPhonon ---
                    if ord < max_order as usize && tau_ext > 2.0 * EPS {
                        let r1 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, 0u32);
                        let r2 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, 2u32);
                        pq = cdf_search(q_cdf, nk, r1 * q_weight_sum) as u32;
                        pnu = uniform_index(r2, nmodes);
                        let q_pdf = q_weight[pq as usize] / (q_weight_sum * f64::cast_from(nmodes));
                        if is_finite_f64(q_pdf) && q_pdf > 0.0 {
                            let r3 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, 4u32);
                            f1 = r3 * tau_ext;
                            let s1 = segment_at(tau, base, ln, f1);
                            if f1 - tau[base + s1] >= DELTA_TAU_LIMIT
                                && tau[base + s1 + 1] - f1 >= DELTA_TAU_LIMIT
                            {
                                let p1 = p_out[base + s1];
                                let mut lambda = phonon_ev[(pq * nmodes + pnu) as usize]
                                    + electron_ev[wrap_sub(grid_n, p1, pq) as usize]
                                    - electron_ev[p1 as usize];
                                if lambda < 1.0e-6 {
                                    lambda = 1.0e-6;
                                }
                                let r4 =
                                    u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, 6u32);
                                f2 = exponential_sample_bounded(r4, lambda, f1, tau_ext);
                                let mut tau_ok = f2 - f1 >= DELTA_TAU_LIMIT;
                                let mut s2 = 0usize;
                                if tau_ok {
                                    s2 = segment_at(tau, base, ln, f2);
                                    tau_ok = f2 - tau[base + s2] >= DELTA_TAU_LIMIT
                                        && tau[base + s2 + 1] - f2 >= DELTA_TAU_LIMIT;
                                }
                                if tau_ok {
                                    let tau_pdf =
                                        exponential_pdf_bounded(f2, lambda, f1, tau_ext) / tau_ext;
                                    // insert_arc_ratio.
                                    let ca_log = changed_arc_factors(
                                        tau, p_out, qv, branch, link, w_re, w_im, v_re, v_im, cbm,
                                        bra_band, long_re, long_im, &mut sm, base, ln, f1, f2, pq,
                                        NONE, NONE, grid_n, rank, nmodes, nbnd, lane, lanes,
                                    );
                                    let ca_pre = sm[SM_PARTIALS as usize + 1];
                                    let ca_pim = sm[SM_PARTIALS as usize + 2];
                                    let seg = segment_change(
                                        tau,
                                        p_out,
                                        electron_ev,
                                        base,
                                        ln,
                                        f1,
                                        f2,
                                        pq,
                                        grid_n,
                                    );
                                    let fr = arc_factor(
                                        w_re,
                                        w_im,
                                        v_re,
                                        v_im,
                                        cbm,
                                        bra_band,
                                        long_re,
                                        long_im,
                                        &mut sm,
                                        wrap_sub(grid_n, p_out[base + s1], pq),
                                        wrap_sub(grid_n, p_out[base + s2], pq),
                                        pq,
                                        pnu,
                                        grid_n,
                                        rank,
                                        nmodes,
                                        nbnd,
                                        lane,
                                        lanes,
                                    );
                                    let fi = sm[SM_PARTIALS as usize];
                                    let modulus = (fr * fr + fi * fi).sqrt();
                                    if modulus == 0.0 || !is_finite_f64(modulus) {
                                        let mut fzero = 0.0f64;
                                        fzero *= 1.0;
                                        ratio_log = 0.0 - 1.0 / fzero;
                                        ratio_pre = 1.0;
                                        ratio_pim = 0.0;
                                    } else {
                                        // Same += grouping as flat insert_arc_ratio.
                                        ratio_log = seg
                                            + ca_log
                                            + (modulus.ln()
                                                - ln_nq
                                                - phonon_ev[(pq * nmodes + pnu) as usize]
                                                    * (f2 - f1));
                                        let ur = fr / modulus;
                                        let ui = fi / modulus;
                                        ratio_pre = ca_pre * ur - ca_pim * ui;
                                        ratio_pim = ca_pre * ui + ca_pim * ur;
                                    }
                                    probability = ratio_log.exp()
                                        / (f64::cast_from((ord + 1) as u32) * q_pdf * tau_pdf);
                                }
                            }
                        }
                    }
                }

                if upd == 3 {
                    // --- RemovePhonon ---
                    if ord > min_order as usize {
                        let r1 = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, 0u32);
                        let nth = uniform_index(r1, ord as u32) as usize;
                        // n-th arc by left-endpoint position (flat nth_arc).
                        let mut seen = 0usize;
                        let mut left = 0usize;
                        for key in 0..ln {
                            let l = link[base + key];
                            if l != NONE && (l as usize) > key {
                                if seen == nth {
                                    left = key;
                                }
                                seen += 1;
                            }
                        }
                        let right = link[base + left] as usize;
                        a_left = left;
                        a_right = right;
                        pq = qv[base + left];
                        pnu = branch[base + left];
                        let q_pdf = q_weight[pq as usize] / (q_weight_sum * f64::cast_from(nmodes));
                        if is_finite_f64(q_pdf) && q_pdf > 0.0 {
                            let p_in = p_out[base + left - 1];
                            let mut lambda = phonon_ev[(pq * nmodes + pnu) as usize]
                                + electron_ev[wrap_sub(grid_n, p_in, pq) as usize]
                                - electron_ev[p_in as usize];
                            if lambda < 1.0e-6 {
                                lambda = 1.0e-6;
                            }
                            let tau_pdf = exponential_pdf_bounded(
                                tau[base + right],
                                lambda,
                                tau[base + left],
                                tau_ext,
                            ) / tau_ext;
                            // remove_arc_ratio: the removed arc's factor first;
                            // |F| = 0 short-circuits to +inf (forced escape).
                            let fr = arc_factor(
                                w_re,
                                w_im,
                                v_re,
                                v_im,
                                cbm,
                                bra_band,
                                long_re,
                                long_im,
                                &mut sm,
                                p_out[base + left],
                                p_out[base + right - 1],
                                pq,
                                pnu,
                                grid_n,
                                rank,
                                nmodes,
                                nbnd,
                                lane,
                                lanes,
                            );
                            let fi = sm[SM_PARTIALS as usize];
                            let modulus = (fr * fr + fi * fi).sqrt();
                            if modulus == 0.0 || !is_finite_f64(modulus) {
                                let mut fzero = 0.0f64;
                                fzero *= 1.0;
                                ratio_log = 1.0 / fzero;
                                ratio_pre = 1.0;
                                ratio_pim = 0.0;
                            } else {
                                let q_neg = neg_idx(grid_n, pq);
                                let ca_log = changed_arc_factors(
                                    tau,
                                    p_out,
                                    qv,
                                    branch,
                                    link,
                                    w_re,
                                    w_im,
                                    v_re,
                                    v_im,
                                    cbm,
                                    bra_band,
                                    long_re,
                                    long_im,
                                    &mut sm,
                                    base,
                                    ln,
                                    tau[base + left],
                                    tau[base + right],
                                    q_neg,
                                    left as u32,
                                    right as u32,
                                    grid_n,
                                    rank,
                                    nmodes,
                                    nbnd,
                                    lane,
                                    lanes,
                                );
                                let ca_pre = sm[SM_PARTIALS as usize + 1];
                                let ca_pim = sm[SM_PARTIALS as usize + 2];
                                let seg = segment_change(
                                    tau,
                                    p_out,
                                    electron_ev,
                                    base,
                                    ln,
                                    tau[base + left],
                                    tau[base + right],
                                    q_neg,
                                    grid_n,
                                );
                                // Same += grouping as flat remove_arc_ratio.
                                ratio_log = seg
                                    + (ca_log - modulus.ln()
                                        + ln_nq
                                        + phonon_ev[(pq * nmodes + pnu) as usize]
                                            * (tau[base + right] - tau[base + left]));
                                // phase = changed · (F/|F|)* .
                                let ur = fr / modulus;
                                let ui = -(fi / modulus);
                                ratio_pre = ca_pre * ur - ca_pim * ui;
                                ratio_pim = ca_pre * ui + ca_pim * ur;
                            }
                            probability =
                                ratio_log.exp() * f64::cast_from(ord as u32) * q_pdf * tau_pdf;
                        }
                    }
                }

                // Metropolis decision (mirrors flat step_selected): impossible
                // draws nothing; certain accepts draw nothing; otherwise one
                // uniform. The word index is the first unread word per update:
                // ChangeTau 2, ChangeInternalTau 4, AddPhonon 8, RemovePhonon 2.
                let mut accepted = false;
                if probability >= 0.0 {
                    if probability >= 1.0 {
                        accepted = true;
                    } else {
                        let mut word = 2u32;
                        if upd == 1 {
                            word = 4u32;
                        }
                        if upd == 2 {
                            word = 8u32;
                        }
                        let r = u01(seed_lo, seed_hi, id_lo, id_hi, step_lo, step_hi, word);
                        if r < probability {
                            accepted = true;
                        }
                    }
                }

                if accepted {
                    // All lanes hold identical proposal registers; lane 0
                    // mutates between barriers.
                    sync_cube();
                    if lane == 0 {
                        if upd == 0 {
                            tau[base + tail] = f1;
                        }
                        if upd == 1 {
                            tau[base + a_left] = f1;
                        }
                        if upd == 2 {
                            // insert_arc(f1, f2, pq, pnu) — flat::FlatDiagram.
                            let mut ln2 = ln;
                            let left = partition_lt(tau, base, ln2, f1);
                            let mut j = ln2;
                            while j > left {
                                tau[base + j] = tau[base + j - 1];
                                p_out[base + j] = p_out[base + j - 1];
                                qv[base + j] = qv[base + j - 1];
                                branch[base + j] = branch[base + j - 1];
                                link[base + j] = link[base + j - 1];
                                j -= 1;
                            }
                            for key in 0..ln2 + 1 {
                                if key != left
                                    && link[base + key] != NONE
                                    && link[base + key] as usize >= left
                                {
                                    link[base + key] += 1;
                                }
                            }
                            ln2 += 1;
                            tau[base + left] = f1;
                            qv[base + left] = pq;
                            branch[base + left] = pnu;
                            link[base + left] = NONE;

                            let right = partition_lt(tau, base, ln2, f2);
                            let mut j = ln2;
                            while j > right {
                                tau[base + j] = tau[base + j - 1];
                                p_out[base + j] = p_out[base + j - 1];
                                qv[base + j] = qv[base + j - 1];
                                branch[base + j] = branch[base + j - 1];
                                link[base + j] = link[base + j - 1];
                                j -= 1;
                            }
                            for key in 0..ln2 + 1 {
                                if key != right
                                    && link[base + key] != NONE
                                    && link[base + key] as usize >= right
                                {
                                    link[base + key] += 1;
                                }
                            }
                            ln2 += 1;
                            tau[base + right] = f2;
                            qv[base + right] = pq;
                            branch[base + right] = pnu;
                            link[base + left] = right as u32;
                            link[base + right] = left as u32;

                            p_out[base + left] = wrap_sub(grid_n, p_out[base + left - 1], pq);
                            for key in left + 1..right {
                                p_out[base + key] = wrap_sub(grid_n, p_out[base + key], pq);
                            }
                            p_out[base + right] = wrap_add(grid_n, p_out[base + right - 1], pq);
                            len[chain as usize] = ln2 as u32;
                        }
                        if upd == 3 {
                            // remove_arc(a_left, a_right).
                            let mut ln2 = ln;
                            // shift_down(a_right), then shift_down(a_left).
                            for key in a_right..ln2 - 1 {
                                tau[base + key] = tau[base + key + 1];
                                p_out[base + key] = p_out[base + key + 1];
                                qv[base + key] = qv[base + key + 1];
                                branch[base + key] = branch[base + key + 1];
                                link[base + key] = link[base + key + 1];
                            }
                            ln2 -= 1;
                            for key in 0..ln2 {
                                let l = link[base + key];
                                if l != NONE {
                                    if l as usize == a_right {
                                        link[base + key] = NONE;
                                    } else if l as usize > a_right {
                                        link[base + key] -= 1;
                                    }
                                }
                            }
                            for key in a_left..ln2 - 1 {
                                tau[base + key] = tau[base + key + 1];
                                p_out[base + key] = p_out[base + key + 1];
                                qv[base + key] = qv[base + key + 1];
                                branch[base + key] = branch[base + key + 1];
                                link[base + key] = link[base + key + 1];
                            }
                            ln2 -= 1;
                            for key in 0..ln2 {
                                let l = link[base + key];
                                if l != NONE {
                                    if l as usize == a_left {
                                        link[base + key] = NONE;
                                    } else if l as usize > a_left {
                                        link[base + key] -= 1;
                                    }
                                }
                            }
                            for key in a_left..a_right - 1 {
                                p_out[base + key] = wrap_add(grid_n, p_out[base + key], pq);
                            }
                            len[chain as usize] = ln2 as u32;
                        }
                        // apply_ratio: cache update with phase renormalization.
                        log_mod[chain as usize] += ratio_log;
                        let cr = phase_re[chain as usize];
                        let ci = phase_im[chain as usize];
                        let mut nr = cr * ratio_pre - ci * ratio_pim;
                        let mut ni = cr * ratio_pim + ci * ratio_pre;
                        let norm = (nr * nr + ni * ni).sqrt();
                        if norm > 0.0 {
                            nr /= norm;
                            ni /= norm;
                        }
                        phase_re[chain as usize] = nr;
                        phase_im[chain as usize] = ni;
                    }
                    sync_cube();
                }

                if lane == 0 {
                    let slot = (chain * 4 + upd) as usize;
                    stats_proposed[slot] += 1;
                    if probability < 0.0 {
                        stats_impossible[slot] += 1;
                    }
                    if accepted {
                        stats_accepted[slot] += 1;
                    }
                }
            }

            // Per-cycle sample: (τ, E_loc, phase, order) — flat local_energy.
            let ln = len[chain as usize] as usize;
            let tail = ln - 1;
            let ord = (ln - 2) / 2;
            let tau_ext = tau[base + tail];
            let mut electron = 0.0f64;
            for key in 0..tail {
                electron += electron_ev[p_out[base + key] as usize]
                    * (tau[base + key + 1] - tau[base + key]);
            }
            let mut phonon = 0.0f64;
            for key in 0..ln {
                let l = link[base + key];
                if l != NONE && (l as usize) > key {
                    phonon += phonon_ev[(qv[base + key] * nmodes + branch[base + key]) as usize]
                        * (tau[base + l as usize] - tau[base + key]);
                }
            }
            let eloc = (electron + phonon - 2.0 * f64::cast_from(ord as u32)) / tau_ext;
            if lane == 0 {
                let i = (cycle * n_chains + chain) as usize;
                samples_tau[i] = tau_ext;
                samples_eloc[i] = eloc;
                samples_pre[i] = phase_re[chain as usize];
                samples_pim[i] = phase_im[chain as usize];
                samples_order[i] = ord as u32;
            }
        }
    }
}
