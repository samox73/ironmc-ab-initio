//! Phase A1: the vertex-dot microbenchmark (GO/NO-GO).
//!
//! Three `#[cube]` variants of the short-range vertex evaluation
//! `g(k,q,ν) = RY_TO_EV · Σ_g cbm[k+q, band(g)]* · wtab[k,g] · vtab[(q,ν),g]`
//! (`VertexTable::short_indices_capped`, full-rank fast path), each validated
//! elementwise against the CPU oracle:
//!
//! 1. `Scalar` — one eval per thread, per-g `bra_band` lookup (CPU mirror).
//! 2. `Vector` — one eval per thread, planar re/im f32 tables loaded as
//!    `Vector<f32, 4>` lanes; the per-pair `bra_band` step function is hoisted
//!    into `nbnd` contiguous band segments (bra_band is monotone by
//!    construction: pair index = n·nbnd + m and `pair_offsets` is sorted).
//! 3. `Coop` — one eval per 32-lane row of a cube, strided coalesced loads,
//!    shared-memory reduce (plane ops are unsupported on the CubeCL CPU
//!    runtime, and for a 16r-byte streaming eval the reduce cost is noise).
//!
//! Cube-dialect rules: see the notes atop `rmc-frohlich/src/gpu/kernel.rs`
//! (usize indexing, literal inits for mutable vars, statement-form ifs,
//! no tuple returns, scalars passed plain).

use cubecl::prelude::*;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};

use crate::vertex_table::{Tab, VertexTable};

/// `reconstruct::RY_TO_EV`, asserted equal at upload.
const RY2EV: f64 = 13.605698066;
/// Lanes per cooperative eval (one CUDA warp) and rows (evals) per cube.
const COOP_LANES: u32 = 32;
const COOP_ROWS: u32 = 4;
/// f32 lanes per vectorized load (float4 on CUDA).
const VEC: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Scalar,
    Vector,
    Coop,
}

impl Variant {
    pub const ALL: [Variant; 3] = [Variant::Scalar, Variant::Vector, Variant::Coop];

    pub fn name(self) -> &'static str {
        match self {
            Variant::Scalar => "scalar",
            Variant::Vector => "vector",
            Variant::Coop => "coop",
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

#[cube]
fn wrap_add(k: u32, q: u32, n: u32) -> u32 {
    let nn = n * n;
    let i = (k / nn + q / nn) % n;
    let j = (k / n % n + q / n % n) % n;
    let l = (k % n + q % n) % n;
    (i * n + j) * n + l
}

/// Variant 1: per-thread scalar loop, per-g bra lookup — mirrors
/// `short_indices_capped` exactly.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn vertex_dot_scalar(
    wtab: &Array<f32>,
    vtab: &Array<f32>,
    cbm: &Array<f64>,
    bra_band: &Array<u32>,
    ks: &Array<u32>,
    qs: &Array<u32>,
    nus: &Array<u32>,
    out: &mut Array<f64>,
    grid_n: u32,
    rank: u32,
    nmodes: u32,
    nbnd: u32,
    n_evals: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_evals as usize {
        let k = ks[i];
        let q = qs[i];
        let nu = nus[i];
        let kp = wrap_add(k, q, grid_n);
        let u0 = (k * rank) as usize * 2;
        let v0 = ((q * nmodes + nu) * rank) as usize * 2;
        let c0 = (kp * nbnd) as usize * 2;
        let mut acc_re = 0.0f64;
        let mut acc_im = 0.0f64;
        for g in 0..rank as usize {
            let w_re = f64::cast_from(wtab[u0 + 2 * g]);
            let w_im = f64::cast_from(wtab[u0 + 2 * g + 1]);
            let v_re = f64::cast_from(vtab[v0 + 2 * g]);
            let v_im = f64::cast_from(vtab[v0 + 2 * g + 1]);
            let b = bra_band[g] as usize;
            let b_re = cbm[c0 + 2 * b];
            let b_im = cbm[c0 + 2 * b + 1];
            let wv_re = w_re * v_re - w_im * v_im;
            let wv_im = w_re * v_im + w_im * v_re;
            acc_re += b_re * wv_re + b_im * wv_im;
            acc_im += b_re * wv_im - b_im * wv_re;
        }
        out[2 * i] = acc_re * RY2EV;
        out[2 * i + 1] = acc_im * RY2EV;
    }
}

/// Variant 2: per-thread eval over planar re/im tables with `Vector<f32, 4>`
/// loads; band segments hoist the bra factor out of the inner loop. Scalar
/// head/tail at unaligned segment edges read the interleaved arrays.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn vertex_dot_vector(
    w_re: &Array<Vector<f32, Const<VEC>>>,
    w_im: &Array<Vector<f32, Const<VEC>>>,
    v_re: &Array<Vector<f32, Const<VEC>>>,
    v_im: &Array<Vector<f32, Const<VEC>>>,
    wtab: &Array<f32>,
    vtab: &Array<f32>,
    cbm: &Array<f64>,
    band_start: &Array<u32>,
    ks: &Array<u32>,
    qs: &Array<u32>,
    nus: &Array<u32>,
    out: &mut Array<f64>,
    grid_n: u32,
    rank: u32,
    nmodes: u32,
    nbnd: u32,
    n_evals: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_evals as usize {
        let k = ks[i];
        let q = qs[i];
        let nu = nus[i];
        let kp = wrap_add(k, q, grid_n);
        // rank % VEC == 0 (asserted at launch), so the per-eval scalar bases
        // are vector-aligned and only band edges need scalar handling.
        let u0 = (k * rank) as usize;
        let v0 = ((q * nmodes + nu) * rank) as usize;
        let c0 = (kp * nbnd) as usize * 2;
        let mut acc_re = 0.0f64;
        let mut acc_im = 0.0f64;
        for band in 0..nbnd as usize {
            let g_begin = band_start[band] as usize;
            let g_end = band_start[band + 1] as usize;
            let j_begin = (g_begin + VEC - 1) / VEC;
            let mut j_end = g_end / VEC;
            if j_end < j_begin {
                j_end = j_begin;
            }
            let mut s_re = 0.0f64;
            let mut s_im = 0.0f64;

            let mut head_end = j_begin * VEC;
            if head_end > g_end {
                head_end = g_end;
            }
            for g in g_begin..head_end {
                let a_re = f64::cast_from(wtab[2 * (u0 + g)]);
                let a_im = f64::cast_from(wtab[2 * (u0 + g) + 1]);
                let b_re = f64::cast_from(vtab[2 * (v0 + g)]);
                let b_im = f64::cast_from(vtab[2 * (v0 + g) + 1]);
                s_re += a_re * b_re - a_im * b_im;
                s_im += a_re * b_im + a_im * b_re;
            }

            let mut acc_rr = Vector::<f64, Const<VEC>>::new(0.0);
            let mut acc_ii = Vector::<f64, Const<VEC>>::new(0.0);
            let mut acc_ri = Vector::<f64, Const<VEC>>::new(0.0);
            let mut acc_ir = Vector::<f64, Const<VEC>>::new(0.0);
            for j in u0 / VEC + j_begin..u0 / VEC + j_end {
                let wr = Vector::<f64, Const<VEC>>::cast_from(w_re[j]);
                let wi = Vector::<f64, Const<VEC>>::cast_from(w_im[j]);
                let vr = Vector::<f64, Const<VEC>>::cast_from(v_re[j + v0 / VEC - u0 / VEC]);
                let vi = Vector::<f64, Const<VEC>>::cast_from(v_im[j + v0 / VEC - u0 / VEC]);
                acc_rr += wr * vr;
                acc_ii += wi * vi;
                acc_ri += wr * vi;
                acc_ir += wi * vr;
            }
            s_re += f64::cast_from((acc_rr - acc_ii).vector_sum());
            s_im += f64::cast_from((acc_ri + acc_ir).vector_sum());

            let mut tail_begin = j_end * VEC;
            if tail_begin < g_begin {
                tail_begin = g_begin;
            }
            for g in tail_begin..g_end {
                let a_re = f64::cast_from(wtab[2 * (u0 + g)]);
                let a_im = f64::cast_from(wtab[2 * (u0 + g) + 1]);
                let b_re = f64::cast_from(vtab[2 * (v0 + g)]);
                let b_im = f64::cast_from(vtab[2 * (v0 + g) + 1]);
                s_re += a_re * b_re - a_im * b_im;
                s_im += a_re * b_im + a_im * b_re;
            }

            let b_re = cbm[c0 + 2 * band];
            let b_im = cbm[c0 + 2 * band + 1];
            acc_re += b_re * s_re + b_im * s_im;
            acc_im += b_re * s_im - b_im * s_re;
        }
        out[2 * i] = acc_re * RY2EV;
        out[2 * i + 1] = acc_im * RY2EV;
    }
}

/// Variant 3: one eval per 32-lane row (a warp on CUDA), lanes stride g for
/// coalesced strip loads, shared-memory reduce, lane 0 writes.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn vertex_dot_coop(
    wtab: &Array<f32>,
    vtab: &Array<f32>,
    cbm: &Array<f64>,
    bra_band: &Array<u32>,
    ks: &Array<u32>,
    qs: &Array<u32>,
    nus: &Array<u32>,
    out: &mut Array<f64>,
    grid_n: u32,
    rank: u32,
    nmodes: u32,
    nbnd: u32,
    n_evals: u32,
) {
    let lane = UNIT_POS_X as usize;
    let row = UNIT_POS_Y;
    let i = (CUBE_POS_X * COOP_ROWS + row) as usize;
    // 2 f64 partials per unit; sized for COOP_LANES * COOP_ROWS = 128 units.
    let mut sm = SharedMemory::<f64>::new(256usize);
    let slot = (row * COOP_LANES) as usize + lane;

    let mut acc_re = 0.0f64;
    let mut acc_im = 0.0f64;
    if i < n_evals as usize {
        let k = ks[i];
        let q = qs[i];
        let nu = nus[i];
        let kp = wrap_add(k, q, grid_n);
        let u0 = (k * rank) as usize * 2;
        let v0 = ((q * nmodes + nu) * rank) as usize * 2;
        let c0 = (kp * nbnd) as usize * 2;
        let iters = (rank as usize + COOP_LANES as usize - 1) / COOP_LANES as usize;
        for t in 0..iters {
            let g = lane + t * COOP_LANES as usize;
            if g < rank as usize {
                let w_re = f64::cast_from(wtab[u0 + 2 * g]);
                let w_im = f64::cast_from(wtab[u0 + 2 * g + 1]);
                let v_re = f64::cast_from(vtab[v0 + 2 * g]);
                let v_im = f64::cast_from(vtab[v0 + 2 * g + 1]);
                let b = bra_band[g] as usize;
                let b_re = cbm[c0 + 2 * b];
                let b_im = cbm[c0 + 2 * b + 1];
                let wv_re = w_re * v_re - w_im * v_im;
                let wv_im = w_re * v_im + w_im * v_re;
                acc_re += b_re * wv_re + b_im * wv_im;
                acc_im += b_re * wv_im - b_im * wv_re;
            }
        }
    }
    sm[2 * slot] = acc_re;
    sm[2 * slot + 1] = acc_im;
    sync_cube();
    if lane == 0 && i < n_evals as usize {
        let mut s_re = 0.0f64;
        let mut s_im = 0.0f64;
        let base = (row * COOP_LANES) as usize;
        for l in 0..COOP_LANES as usize {
            s_re += sm[2 * (base + l)];
            s_im += sm[2 * (base + l) + 1];
        }
        out[2 * i] = s_re * RY2EV;
        out[2 * i + 1] = s_im * RY2EV;
    }
    // The CubeCL CPU runtime reuses one shared-memory arena across its
    // per-worker cube loop, so the read phase must be fenced off from the
    // next cube's writes; on device this is a per-eval barrier, noise next
    // to the 16r-byte strip stream.
    sync_cube();
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

fn tab_to_f32(tab: &Tab, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(2 * len);
    match tab {
        Tab::F32(values) => {
            for value in values {
                out.push(value.re);
                out.push(value.im);
            }
        }
        Tab::F64(values) => {
            for value in values {
                out.push(value.re as f32);
                out.push(value.im as f32);
            }
        }
    }
    assert_eq!(out.len(), 2 * len);
    out
}

/// Device-resident vertex tables (interleaved f32 + planar f32 copies for the
/// vector variant; grid data f64), plus the dims the kernels need.
pub struct DeviceTables<R: Runtime> {
    pub client: ComputeClient<R>,
    pub grid_n: u32,
    pub rank: u32,
    pub nmodes: u32,
    pub nbnd: u32,
    wtab: cubecl::server::Handle,
    vtab: cubecl::server::Handle,
    w_re: cubecl::server::Handle,
    w_im: cubecl::server::Handle,
    v_re: cubecl::server::Handle,
    v_im: cubecl::server::Handle,
    cbm: cubecl::server::Handle,
    bra_band: cubecl::server::Handle,
    band_start: cubecl::server::Handle,
}

impl<R: Runtime> DeviceTables<R> {
    pub fn upload(client: ComputeClient<R>, table: &VertexTable) -> Self {
        assert_eq!(RY2EV, crate::reconstruct::RY_TO_EV);
        let nk = table.len();
        let rank = table.s.len();
        assert!(rank > 0 && rank % VEC == 0, "rank must be a multiple of {VEC}");
        assert_eq!(table.bra_band.len(), rank, "table must be precontracted");
        assert!(
            table.bra_band.windows(2).all(|w| w[0] <= w[1]),
            "bra_band must be monotone (pair-major component order)"
        );
        // band_start[b] = first component with band >= b; segments are exact
        // because bra_band is monotone.
        let band_start: Vec<u32> = (0..=table.nbnd)
            .map(|b| table.bra_band.partition_point(|&x| (x as usize) < b) as u32)
            .collect();

        let wtab = tab_to_f32(&table.wtab, nk * rank);
        let vtab = tab_to_f32(&table.vtab, nk * table.nmodes * rank);
        let planar = |interleaved: &[f32], re: bool| -> Vec<f32> {
            interleaved
                .chunks_exact(2)
                .map(|c| if re { c[0] } else { c[1] })
                .collect()
        };
        let cbm: Vec<f64> = table
            .cbm_evec
            .iter()
            .flat_map(|z| [z.re, z.im])
            .collect();
        let bra_band: Vec<u32> = table.bra_band.iter().map(|&b| b as u32).collect();

        Self {
            grid_n: table.grid_n as u32,
            rank: rank as u32,
            nmodes: table.nmodes as u32,
            nbnd: table.nbnd as u32,
            wtab: client.create_from_slice(f32::as_bytes(&wtab)),
            vtab: client.create_from_slice(f32::as_bytes(&vtab)),
            w_re: client.create_from_slice(f32::as_bytes(&planar(&wtab, true))),
            w_im: client.create_from_slice(f32::as_bytes(&planar(&wtab, false))),
            v_re: client.create_from_slice(f32::as_bytes(&planar(&vtab, true))),
            v_im: client.create_from_slice(f32::as_bytes(&planar(&vtab, false))),
            cbm: client.create_from_slice(f64::as_bytes(&cbm)),
            bra_band: client.create_from_slice(u32::as_bytes(&bra_band)),
            band_start: client.create_from_slice(u32::as_bytes(&band_start)),
            client,
        }
    }

    /// Launch one variant over the given eval set and return per-eval results.
    /// The second element is the kernel wall time of the *second* (JIT-warm)
    /// launch when `timed` is set, else of the single launch.
    pub fn evaluate(
        &self,
        variant: Variant,
        ks: &[u32],
        qs: &[u32],
        nus: &[u32],
        timed: bool,
    ) -> (Vec<Complex64>, f64) {
        let n = ks.len();
        assert!(n > 0 && qs.len() == n && nus.len() == n);
        let client = &self.client;
        let ks_h = client.create_from_slice(u32::as_bytes(ks));
        let qs_h = client.create_from_slice(u32::as_bytes(qs));
        let nus_h = client.create_from_slice(u32::as_bytes(nus));
        let out_h = client.create_from_slice(f64::as_bytes(&vec![0.0f64; 2 * n]));
        let fence = client.create_from_slice(f64::as_bytes(&[0.0f64]));

        let launches = if timed { 2 } else { 1 };
        let mut secs = 0.0;
        for _ in 0..launches {
            let t0 = std::time::Instant::now();
            self.launch(variant, n, &ks_h, &qs_h, &nus_h, &out_h);
            // A blocking read on the in-order queue fences the launch.
            let _ = f64::from_bytes(&client.read_one_unchecked(fence.clone()))[0];
            secs = t0.elapsed().as_secs_f64();
        }

        let raw = f64::from_bytes(&client.read_one_unchecked(out_h)).to_vec();
        let values = raw
            .chunks_exact(2)
            .map(|c| Complex64::new(c[0], c[1]))
            .collect();
        (values, secs)
    }

    fn launch(
        &self,
        variant: Variant,
        n: usize,
        ks: &cubecl::server::Handle,
        qs: &cubecl::server::Handle,
        nus: &cubecl::server::Handle,
        out: &cubecl::server::Handle,
    ) {
        let client = &self.client;
        let nk = (self.grid_n as usize).pow(3);
        let rank = self.rank as usize;
        let arr = |h: &cubecl::server::Handle, len: usize| -> ArrayArg<R> {
            unsafe { ArrayArg::from_raw_parts(h.clone(), len) }
        };
        let wtab_len = 2 * nk * rank;
        let vtab_len = 2 * nk * self.nmodes as usize * rank;
        let cbm_len = 2 * nk * self.nbnd as usize;
        match variant {
            Variant::Scalar => vertex_dot_scalar::launch::<R>(
                client,
                CubeCount::Static((n as u32).div_ceil(256), 1, 1),
                CubeDim::new_1d(256),
                arr(&self.wtab, wtab_len),
                arr(&self.vtab, vtab_len),
                arr(&self.cbm, cbm_len),
                arr(&self.bra_band, rank),
                arr(ks, n),
                arr(qs, n),
                arr(nus, n),
                arr(out, 2 * n),
                self.grid_n,
                self.rank,
                self.nmodes,
                self.nbnd,
                n as u32,
            ),
            Variant::Vector => vertex_dot_vector::launch::<R>(
                client,
                CubeCount::Static((n as u32).div_ceil(256), 1, 1),
                CubeDim::new_1d(256),
                arr(&self.w_re, wtab_len / 2 / VEC),
                arr(&self.w_im, wtab_len / 2 / VEC),
                arr(&self.v_re, vtab_len / 2 / VEC),
                arr(&self.v_im, vtab_len / 2 / VEC),
                arr(&self.wtab, wtab_len),
                arr(&self.vtab, vtab_len),
                arr(&self.cbm, cbm_len),
                arr(&self.band_start, self.nbnd as usize + 1),
                arr(ks, n),
                arr(qs, n),
                arr(nus, n),
                arr(out, 2 * n),
                self.grid_n,
                self.rank,
                self.nmodes,
                self.nbnd,
                n as u32,
            ),
            Variant::Coop => vertex_dot_coop::launch::<R>(
                client,
                CubeCount::Static((n as u32).div_ceil(COOP_ROWS), 1, 1),
                CubeDim::new_2d(COOP_LANES, COOP_ROWS),
                arr(&self.wtab, wtab_len),
                arr(&self.vtab, vtab_len),
                arr(&self.cbm, cbm_len),
                arr(&self.bra_band, rank),
                arr(ks, n),
                arr(qs, n),
                arr(nus, n),
                arr(out, 2 * n),
                self.grid_n,
                self.rank,
                self.nmodes,
                self.nbnd,
                n as u32,
            ),
        }
    }
}

/// Random eval set mirroring MC access: k uniform over the grid, q uniform
/// over non-Γ points, ν uniform over branches.
pub fn random_evals(table: &VertexTable, n: usize, seed: u64) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let nk = table.len();
    let ks = (0..n).map(|_| rng.gen_range(0..nk) as u32).collect();
    let qs = (0..n).map(|_| rng.gen_range(1..nk) as u32).collect();
    let nus = (0..n)
        .map(|_| rng.gen_range(0..table.nmodes) as u32)
        .collect();
    (ks, qs, nus)
}

/// Worst |device − oracle| over the eval set, relative to the oracle's scale
/// (mirrors `VertexTable::max_direct_error`).
pub fn max_rel_error(
    table: &VertexTable,
    ks: &[u32],
    qs: &[u32],
    nus: &[u32],
    values: &[Complex64],
) -> f64 {
    let mut worst: f64 = 0.0;
    let mut scale: f64 = 0.0;
    for (((&k, &q), &nu), &value) in ks.iter().zip(qs).zip(nus).zip(values) {
        let expected =
            table.short_indices_capped(k as usize, q as usize, nu as usize, &usize::MAX);
        scale = scale.max(expected.norm());
        worst = worst.max((value - expected).norm());
    }
    worst / scale
}

/// USER CHECKPOINT driver: correctness on a subsample, then the timed sweep.
/// Prints per-variant evals/s and table-stream GB/s (16·rank bytes per eval —
/// DRAM-side wtab+vtab strips only; bra/index/output traffic is cache-resident
/// or negligible).
#[cfg(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda"))]
pub fn run_vertex_bench(
    table: &VertexTable,
    n_evals: usize,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::gpu::SelectedRuntime;

    let client = <SelectedRuntime as Runtime>::client(&Default::default());
    let t0 = std::time::Instant::now();
    let tables = DeviceTables::upload(client, table);
    println!(
        "upload: grid {n}³, rank {r}, nmodes {m}, {s:.3} s",
        n = table.grid_n,
        r = table.s.len(),
        m = table.nmodes,
        s = t0.elapsed().as_secs_f64()
    );

    let (cks, cqs, cnus) = random_evals(table, 512, seed ^ 0xc0ffee);
    let bytes_per_eval = 16.0 * table.s.len() as f64;
    let (ks, qs, nus) = random_evals(table, n_evals, seed);

    // Host baseline: the GO/NO-GO compares device evals/s against the CPU
    // node's measured throughput on the same eval set (all cores, rayon).
    {
        use rayon::prelude::*;
        let t0 = std::time::Instant::now();
        let sum: f64 = (0..n_evals)
            .into_par_iter()
            .map(|i| {
                table
                    .short_indices_capped(
                        ks[i] as usize,
                        qs[i] as usize,
                        nus[i] as usize,
                        &usize::MAX,
                    )
                    .re
            })
            .sum();
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "cpu-host {evals:>9.3e} evals/s  {gbs:>7.1} GB/s (table stream)  \
             [{threads} threads, {n_evals} evals, {secs:.3} s, checksum {sum:.3e}]",
            evals = n_evals as f64 / secs,
            gbs = n_evals as f64 * bytes_per_eval / secs / 1.0e9,
            threads = rayon::current_num_threads(),
        );
    }

    for variant in Variant::ALL {
        let (values, _) = tables.evaluate(variant, &cks, &cqs, &cnus, false);
        let err = max_rel_error(table, &cks, &cqs, &cnus, &values);
        assert!(
            err <= 1.0e-6,
            "variant {} exceeds 1e-6 relative: {err:.3e}",
            variant.name()
        );
        let (_, secs) = tables.evaluate(variant, &ks, &qs, &nus, true);
        println!(
            "{name:<7} {evals:>9.3e} evals/s  {gbs:>7.1} GB/s (table stream)  \
             rel_err {err:.2e}  [{n_evals} evals, {secs:.3} s]",
            name = variant.name(),
            evals = n_evals as f64 / secs,
            gbs = n_evals as f64 * bytes_per_eval / secs / 1.0e9,
        );
    }
    Ok(())
}
