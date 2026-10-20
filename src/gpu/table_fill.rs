//! GPU Fourier fill for `VertexTable::build_range`. Replaces the CPU
//! `fill_fourier_tables` with three hand-rolled reductions on the device,
//! matching the CPU math exactly up to summation order (validated within f32
//! tolerance against the CPU fill — see the parity test in `vertex_table.rs`).
//! Base data stays on the CPU; only the fill moves to the GPU.
//!
//! The fill, per grid point, is two GEMM-shaped reductions:
//!   wtab[p][g]      = Σ_re PE[p][re] · U_w[re][g]
//!   vtab[p][ν][g]   = Σ_a  evec[p][ν][a] · B[p][a][g],
//!                     B[p][a][g] = Σ_rp PP[p][rp] · V_w[rp][a][g]
//! with U_w[re][g] = u_at(g,re)/ndegen_k(re, n_g, m_g),
//!      V_w[rp][a][g] = v_at(g,rp,a)/ndegen_g(rp, 0, a/3),
//!      PE[p][re] = exp(i 2π point_p·irvec_k[re]),  PP analogous over irvec_g.
//! U_w/V_w are shard-invariant (uploaded once); PE/PP/evec are per-chunk.
//!
//! Complex is stored planar (separate re/im arrays), f64 on the compute side,
//! f32 in the stored output — the CPU fill likewise accumulates in f64 and rounds
//! to f32, so the only difference from the CPU table is summation order. Cube
//! dialect: scalar per-output-element threads, no plane/shared-memory ops (so CPU
//! and CUDA runtimes behave identically), and the four macro landmines from the
//! frohlich port are avoided (literal-only mutable inits, no literal `if`-selects).

use core::mem::size_of;

use cubecl::prelude::*;
use num_complex::{Complex, Complex64};
use rayon::prelude::*;

use crate::eph_svd::EphSvd;
use crate::epw::Wigner;
use crate::reconstruct::phase;
use crate::vertex_table::VertexTable;

/// Grid points processed per kernel launch. The device-resident B intermediate is
/// `chunk · nmodes · rank` complex f64 (≈ 1.3 GB at chunk=16384, rank=800), well
/// within an A100's per-allocation cap. Override via `fill_fourier_gpu`'s arg.
pub const DEFAULT_GPU_CHUNK: usize = 16384;

/// `wtab[p][g] = Σ_re PE[p][re] · U_w[re][g]` — one thread per (point, component).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn fill_wtab_kernel(
    pe_re: &Array<f64>,
    pe_im: &Array<f64>,
    uw_re: &Array<f64>,
    uw_im: &Array<f64>,
    wtab_re: &mut Array<f32>,
    wtab_im: &mut Array<f32>,
    n_out: u32,
    nrr_k: u32,
    comp: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_out as usize {
        let g = i % comp as usize;
        let p = i / comp as usize;
        let mut acc_re = 0.0f64;
        let mut acc_im = 0.0f64;
        let pe_base = p * nrr_k as usize;
        for r in 0..nrr_k as usize {
            let ar = pe_re[pe_base + r];
            let ai = pe_im[pe_base + r];
            let br = uw_re[r * comp as usize + g];
            let bi = uw_im[r * comp as usize + g];
            acc_re += ar * br - ai * bi;
            acc_im += ar * bi + ai * br;
        }
        wtab_re[i] = f32::cast_from(acc_re);
        wtab_im[i] = f32::cast_from(acc_im);
    }
}

/// `B[p][a][g] = Σ_rp PP[p][rp] · V_w[rp][a][g]` — one thread per (point, cart, component).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn fill_b_kernel(
    pp_re: &Array<f64>,
    pp_im: &Array<f64>,
    vw_re: &Array<f64>,
    vw_im: &Array<f64>,
    b_re: &mut Array<f64>,
    b_im: &mut Array<f64>,
    n_out: u32,
    nrr_g: u32,
    nm: u32,
    comp: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_out as usize {
        let g = i % comp as usize;
        let pa = i / comp as usize; // p*nm + a
        let a = pa % nm as usize;
        let p = pa / nm as usize;
        let mut acc_re = 0.0f64;
        let mut acc_im = 0.0f64;
        let pp_base = p * nrr_g as usize;
        for rp in 0..nrr_g as usize {
            let ar = pp_re[pp_base + rp];
            let ai = pp_im[pp_base + rp];
            let vidx = (rp * nm as usize + a) * comp as usize + g;
            let br = vw_re[vidx];
            let bi = vw_im[vidx];
            acc_re += ar * br - ai * bi;
            acc_im += ar * bi + ai * br;
        }
        b_re[i] = acc_re;
        b_im[i] = acc_im;
    }
}

/// `vtab[p][ν][g] = Σ_a evec[p][ν][a] · B[p][a][g]` — one thread per (point, branch, component).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn fill_vtab_kernel(
    evec_re: &Array<f64>,
    evec_im: &Array<f64>,
    b_re: &Array<f64>,
    b_im: &Array<f64>,
    vtab_re: &mut Array<f32>,
    vtab_im: &mut Array<f32>,
    n_out: u32,
    nm: u32,
    comp: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_out as usize {
        let g = i % comp as usize;
        let pnu = i / comp as usize; // p*nm + nu
        let nu = pnu % nm as usize;
        let p = pnu / nm as usize;
        let mut acc_re = 0.0f64;
        let mut acc_im = 0.0f64;
        let evec_base = (p * nm as usize + nu) * nm as usize;
        for a in 0..nm as usize {
            let er = evec_re[evec_base + a];
            let ei = evec_im[evec_base + a];
            let bidx = (p * nm as usize + a) * comp as usize + g;
            let br = b_re[bidx];
            let bi = b_im[bidx];
            acc_re += er * br - ei * bi;
            acc_im += er * bi + ei * br;
        }
        vtab_re[i] = f32::cast_from(acc_re);
        vtab_im[i] = f32::cast_from(acc_im);
    }
}

/// `U_w[re][g] = u_at(g,re) / ndegen_k(re, n_g, m_g)` (0 where ndegen is 0), planar.
fn build_uw(
    svd: &EphSvd,
    wigner: &Wigner,
    component_bands: &[(usize, usize)],
) -> (Vec<f64>, Vec<f64>) {
    let (nrr_k, comp) = (svd.nrr_k, svd.rank());
    let mut re = vec![0.0f64; nrr_k * comp];
    let mut im = vec![0.0f64; nrr_k * comp];
    for g in 0..comp {
        let (n, m) = component_bands[g];
        for r in 0..nrr_k {
            let d = wigner.k.ndegen(r, n, m);
            if d > 0 {
                let val = svd.u_at(g, r) / d as f64;
                re[r * comp + g] = val.re;
                im[r * comp + g] = val.im;
            }
        }
    }
    (re, im)
}

/// `V_w[rp][a][g] = v_at(g,rp,a) / ndegen_g(rp, 0, a/3)` (0 where ndegen is 0), planar.
fn build_vw(svd: &EphSvd, wigner: &Wigner) -> (Vec<f64>, Vec<f64>) {
    let (nrr_g, nm, comp) = (svd.nrr_g, svd.nmodes, svd.rank());
    let mut re = vec![0.0f64; nrr_g * nm * comp];
    let mut im = vec![0.0f64; nrr_g * nm * comp];
    for g in 0..comp {
        for rp in 0..nrr_g {
            for a in 0..nm {
                let d = wigner.g.ndegen(rp, 0, a / 3);
                if d > 0 {
                    let val = svd.v_at(g, rp, a) / d as f64;
                    re[(rp * nm + a) * comp + g] = val.re;
                    im[(rp * nm + a) * comp + g] = val.im;
                }
            }
        }
    }
    (re, im)
}

/// `PE[p][r] = phase(point(start+p), irvec[r])` for a chunk of `np` points, planar.
fn phases(grid_n: usize, start: usize, np: usize, irvec: &[[i64; 3]]) -> (Vec<f64>, Vec<f64>) {
    let nrr = irvec.len();
    let mut re = vec![0.0f64; np * nrr];
    let mut im = vec![0.0f64; np * nrr];
    re.par_chunks_mut(nrr)
        .zip(im.par_chunks_mut(nrr))
        .enumerate()
        .for_each(|(p, (rr, ii))| {
            let pt = VertexTable::point_for(grid_n, start + p);
            for (r, &rv) in irvec.iter().enumerate() {
                let ph = phase(pt, rv);
                rr[r] = ph.re;
                ii[r] = ph.im;
            }
        });
    (re, im)
}

/// Split a chunk of the phonon-eigenvector slab into planar re/im arrays.
fn planar_evec(phonon_evec: &[Complex64], c0: usize, np: usize, nm: usize) -> (Vec<f64>, Vec<f64>) {
    let n = np * nm * nm;
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    for j in 0..n {
        let z = phonon_evec[c0 * nm * nm + j];
        re[j] = z.re;
        im[j] = z.im;
    }
    (re, im)
}

/// Fill the `[lo, hi)` slab of `wtab`/`vtab` on the device. `wtab` is `[point][g]`,
/// `vtab` is `[point][branch][component]`; both are the slab's rows (length
/// `npoints·rank` and `npoints·nmodes·rank`). `phonon_evec` is the slab's
/// eigenvectors `[point][branch][cart]`. Generic over the cubecl runtime so the
/// same code runs on CUDA (production) and the CPU runtime (parity test).
#[allow(clippy::too_many_arguments)]
pub fn fill_fourier_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    wigner: &Wigner,
    svd: &EphSvd,
    grid_n: usize,
    lo: usize,
    hi: usize,
    phonon_evec: &[Complex64],
    component_bands: &[(usize, usize)],
    wtab: &mut [Complex<f32>],
    vtab: &mut [Complex<f32>],
    chunk_points: usize,
) {
    let (nm, nrr_k, nrr_g, comp) = (svd.nmodes, svd.nrr_k, svd.nrr_g, svd.rank());
    let npoints = hi - lo;
    if comp == 0 || npoints == 0 {
        return;
    }
    let chunk = chunk_points.max(1);
    const BLOCK: u32 = 256;

    // Shard-invariant factors: build on host, upload once.
    let (uw_re, uw_im) = build_uw(svd, wigner, component_bands);
    let (vw_re, vw_im) = build_vw(svd, wigner);
    let uw_re_h = client.create_from_slice(f64::as_bytes(&uw_re));
    let uw_im_h = client.create_from_slice(f64::as_bytes(&uw_im));
    let vw_re_h = client.create_from_slice(f64::as_bytes(&vw_re));
    let vw_im_h = client.create_from_slice(f64::as_bytes(&vw_im));

    let arr = |h: &cubecl::server::Handle, len: usize| -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(h.clone(), len) }
    };

    log::info!(
        "GPU Fourier fill: {npoints} points, rank {comp}, nrr_k {nrr_k}, nrr_g {nrr_g}, \
         chunk {chunk} (B intermediate {:.2} GiB/chunk)",
        (chunk * nm * comp * 2 * size_of::<f64>()) as f64 / 2_f64.powi(30)
    );

    // Kernel sizes and launch counts are u32 on the device; a huge $RMC_GPU_CHUNK
    // would wrap silently and leave most of the output buffers uninitialized.
    assert!(
        chunk * nm * comp <= u32::MAX as usize,
        "GPU chunk too large: chunk·nmodes·rank must fit u32 — lower $RMC_GPU_CHUNK"
    );

    let mut c0 = 0;
    while c0 < npoints {
        let np = chunk.min(npoints - c0);
        let (pe_re, pe_im) = phases(grid_n, lo + c0, np, &wigner.k.irvec);
        let (pp_re, pp_im) = phases(grid_n, lo + c0, np, &wigner.g.irvec);
        let (ev_re, ev_im) = planar_evec(phonon_evec, c0, np, nm);
        let pe_re_h = client.create_from_slice(f64::as_bytes(&pe_re));
        let pe_im_h = client.create_from_slice(f64::as_bytes(&pe_im));
        let pp_re_h = client.create_from_slice(f64::as_bytes(&pp_re));
        let pp_im_h = client.create_from_slice(f64::as_bytes(&pp_im));
        let ev_re_h = client.create_from_slice(f64::as_bytes(&ev_re));
        let ev_im_h = client.create_from_slice(f64::as_bytes(&ev_im));

        // wtab chunk.
        let wn = np * comp;
        let wtab_re_h = client.empty(wn * size_of::<f32>());
        let wtab_im_h = client.empty(wn * size_of::<f32>());
        fill_wtab_kernel::launch::<R>(
            client,
            CubeCount::Static((wn as u32).div_ceil(BLOCK), 1, 1),
            CubeDim::new_1d(BLOCK),
            arr(&pe_re_h, np * nrr_k),
            arr(&pe_im_h, np * nrr_k),
            arr(&uw_re_h, nrr_k * comp),
            arr(&uw_im_h, nrr_k * comp),
            arr(&wtab_re_h, wn),
            arr(&wtab_im_h, wn),
            wn as u32,
            nrr_k as u32,
            comp as u32,
        );
        let wr = f32::from_bytes(&client.read_one_unchecked(wtab_re_h)).to_vec();
        let wi = f32::from_bytes(&client.read_one_unchecked(wtab_im_h)).to_vec();
        for j in 0..wn {
            wtab[c0 * comp + j] = Complex::new(wr[j], wi[j]);
        }

        // vtab chunk: B (stage 1, device-resident) then contract with evec (stage 2).
        let vn = np * nm * comp;
        let b_re_h = client.empty(vn * size_of::<f64>());
        let b_im_h = client.empty(vn * size_of::<f64>());
        fill_b_kernel::launch::<R>(
            client,
            CubeCount::Static((vn as u32).div_ceil(BLOCK), 1, 1),
            CubeDim::new_1d(BLOCK),
            arr(&pp_re_h, np * nrr_g),
            arr(&pp_im_h, np * nrr_g),
            arr(&vw_re_h, nrr_g * nm * comp),
            arr(&vw_im_h, nrr_g * nm * comp),
            arr(&b_re_h, vn),
            arr(&b_im_h, vn),
            vn as u32,
            nrr_g as u32,
            nm as u32,
            comp as u32,
        );
        let vtab_re_h = client.empty(vn * size_of::<f32>());
        let vtab_im_h = client.empty(vn * size_of::<f32>());
        fill_vtab_kernel::launch::<R>(
            client,
            CubeCount::Static((vn as u32).div_ceil(BLOCK), 1, 1),
            CubeDim::new_1d(BLOCK),
            arr(&ev_re_h, np * nm * nm),
            arr(&ev_im_h, np * nm * nm),
            arr(&b_re_h, vn),
            arr(&b_im_h, vn),
            arr(&vtab_re_h, vn),
            arr(&vtab_im_h, vn),
            vn as u32,
            nm as u32,
            comp as u32,
        );
        let vr = f32::from_bytes(&client.read_one_unchecked(vtab_re_h)).to_vec();
        let vi = f32::from_bytes(&client.read_one_unchecked(vtab_im_h)).to_vec();
        for j in 0..vn {
            vtab[c0 * nm * comp + j] = Complex::new(vr[j], vi[j]);
        }
        c0 += np;
    }
}

/// Non-generic entry: create the selected-runtime client and fill (f32 output).
/// Gated on a concrete backend — the `gpu` feature alone has no runtime.
#[cfg(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn fill_fourier_gpu_f32(
    wigner: &Wigner,
    svd: &EphSvd,
    grid_n: usize,
    lo: usize,
    hi: usize,
    phonon_evec: &[Complex64],
    component_bands: &[(usize, usize)],
    wtab: &mut [Complex<f32>],
    vtab: &mut [Complex<f32>],
    chunk_points: usize,
) {
    let client = <crate::gpu::SelectedRuntime as Runtime>::client(&Default::default());
    fill_fourier_gpu::<crate::gpu::SelectedRuntime>(
        &client,
        wigner,
        svd,
        grid_n,
        lo,
        hi,
        phonon_evec,
        component_bands,
        wtab,
        vtab,
        chunk_points,
    );
}
