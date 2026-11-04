//! Uniform-grid Fourier tables for the SVD-compressed vertex.
//!
//! A table may have a global production budget from
//! [`EphSvd::compress_global`]. The `rank_cap` arguments below are per-pair caps
//! retained only for cheap sweeps of an already-built table.

use crate::eph_svd::EphSvd;
use crate::epw::{Crystal, EpwData, Wigner};
use crate::longrange::{g_long_cart, project_cartesian_branches, quantized_eigenvectors};
use crate::reconstruct::{interpolate_d_polar, interpolate_h, phase, RY_TO_EV, RY_TO_THZ};
use crate::vertex_direct::DirectVertex;
use nalgebra::Vector3;
use num_complex::{Complex, Complex64};
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

type Cf = Complex64;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub const DEFAULT_PHFREQ_CUTOFF_EV: f64 = 1.0e-3;
/// Sized for the VSC-5 512 GB nodes (zen3_0512 incl. devel, a100x2) with OS
/// headroom; n=120 at rank 800/f32 is ~72 GiB. Override via --max-gib or
/// $RMC_TABLE_MAX_GIB on smaller machines.
pub const DEFAULT_MAX_BYTES: u64 = 480 * 1024_u64.pow(3);

const CACHE_MAGIC: &[u8; 8] = b"RMCVTAB\0";
const CACHE_VERSION: u32 = 1;
const FLAG_F32: u32 = 1;
/// Set on shard files so a shard can never be misread as a full table (or vice
/// versa) — without the bit, both formats share the header and a cross-load
/// only fails deep in bincode with an opaque error.
const FLAG_SHARD: u32 = 2;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TablePrecision {
    F32,
    F64,
}

impl TablePrecision {
    fn bytes(self) -> u64 {
        match self {
            Self::F32 => 8,
            Self::F64 => 16,
        }
    }
}

impl std::str::FromStr for TablePrecision {
    type Err = &'static str;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            _ => Err("precision must be f32 or f64"),
        }
    }
}

/// Which engine fills the Fourier tables. Base data is always CPU; only the fill
/// (the build's dominant cost) can move to the GPU. GPU output is f32 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FillDevice {
    #[default]
    Cpu,
    /// `chunk_points`: grid points per kernel launch (see [`DEFAULT_GPU_CHUNK`]).
    Gpu { chunk_points: usize },
}

/// Grid points processed per GPU kernel launch. The device-resident B intermediate is
/// `chunk · nmodes · rank` complex f64 (≈ 1.3 GB at chunk=16384, rank=800), well
/// within an A100's per-allocation cap. The CLI lowers it via `$RMC_GPU_CHUNK`.
pub const DEFAULT_GPU_CHUNK: usize = 16384;

impl std::str::FromStr for FillDevice {
    type Err = &'static str;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu {
                chunk_points: DEFAULT_GPU_CHUNK,
            }),
            _ => Err("device must be cpu or gpu"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Tab {
    F32(Vec<Complex<f32>>),
    F64(Vec<Complex64>),
}

impl Tab {
    fn zeros(len: usize, precision: TablePrecision) -> Self {
        match precision {
            TablePrecision::F32 => Self::F32(vec![Complex::new(0.0, 0.0); len]),
            TablePrecision::F64 => Self::F64(vec![Complex64::new(0.0, 0.0); len]),
        }
    }

    #[inline]
    fn get(&self, index: usize) -> Complex64 {
        match self {
            Self::F32(values) => {
                let value = values[index];
                Complex64::new(value.re as f64, value.im as f64)
            }
            Self::F64(values) => values[index],
        }
    }

    fn precision(&self) -> TablePrecision {
        match self {
            Self::F32(_) => TablePrecision::F32,
            Self::F64(_) => TablePrecision::F64,
        }
    }

    fn scale(&mut self, index: usize, factor: Complex64) {
        match self {
            Self::F32(values) => {
                let value = Cf::new(values[index].re as f64, values[index].im as f64) * factor;
                values[index] = Complex::new(value.re as f32, value.im as f32);
            }
            Self::F64(values) => values[index] *= factor,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// Concatenate same-precision parts in order (shard reassembly). Exact
    /// preallocation — a growth-doubling realloc at n=120 would transiently
    /// hold an extra copy of a ~70 GiB table.
    fn concat<'a>(parts: impl Iterator<Item = &'a Tab>, precision: TablePrecision) -> Result<Tab> {
        let parts: Vec<&Tab> = parts.collect();
        let total: usize = parts.iter().map(|part| part.len()).sum();
        match precision {
            TablePrecision::F32 => {
                let mut out = Vec::with_capacity(total);
                for part in parts {
                    match part {
                        Tab::F32(values) => out.extend_from_slice(values),
                        Tab::F64(_) => return Err("mixed-precision shards".into()),
                    }
                }
                Ok(Tab::F32(out))
            }
            TablePrecision::F64 => {
                let mut out = Vec::with_capacity(total);
                for part in parts {
                    match part {
                        Tab::F64(values) => out.extend_from_slice(values),
                        Tab::F32(_) => return Err("mixed-precision shards".into()),
                    }
                }
                Ok(Tab::F64(out))
            }
        }
    }
}

/// Shard metadata: which balanced slab of the `grid_n³` grid this file holds,
/// plus the build parameters (SVD budget, precision) so the resume check can
/// reject a stale shard from a different build, not only a different slab.
/// Written ahead of the slab table so a resume check can read it cheaply.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardMeta {
    pub grid_n: usize,
    pub shards: usize,
    pub shard: usize,
    pub lo: usize,
    pub hi: usize,
    pub total_rank: usize,
    pub precision: TablePrecision,
}

fn write_cache_header(
    writer: &mut impl Write,
    precision: TablePrecision,
    is_shard: bool,
) -> Result<()> {
    writer.write_all(CACHE_MAGIC)?;
    writer.write_all(&CACHE_VERSION.to_le_bytes())?;
    let mut flags = match precision {
        TablePrecision::F32 => FLAG_F32,
        TablePrecision::F64 => 0,
    };
    if is_shard {
        flags |= FLAG_SHARD;
    }
    writer.write_all(&flags.to_le_bytes())?;
    Ok(())
}

fn read_cache_header(reader: &mut impl Read) -> Result<(TablePrecision, bool)> {
    let mut magic = [0; 8];
    let mut version = [0; 4];
    let mut flags = [0; 4];
    reader.read_exact(&mut magic)?;
    reader.read_exact(&mut version)?;
    reader.read_exact(&mut flags)?;
    if &magic != CACHE_MAGIC {
        return Err("invalid vertex-table cache; rebuild the table".into());
    }
    let version = u32::from_le_bytes(version);
    if version != CACHE_VERSION {
        return Err(format!(
            "vertex-table cache version {version} is unsupported (expected {CACHE_VERSION}); rebuild the table"
        )
        .into());
    }
    let flags = u32::from_le_bytes(flags);
    if flags & !(FLAG_F32 | FLAG_SHARD) != 0 {
        return Err("unsupported vertex-table cache flags; rebuild the table".into());
    }
    let precision = if flags & FLAG_F32 != 0 { TablePrecision::F32 } else { TablePrecision::F64 };
    Ok((precision, flags & FLAG_SHARD != 0))
}

/// E⁽²⁾ = (1/N_q) Σ_{q≠0,ν} |g_ν(0,q)|² / (ε(0)−ε(−q)−ω_ν(q)).
///
/// `ε(0)` is kept explicitly: the table shifts energies to the global band
/// minimum, so `ε(Γ) > 0` for a material whose minimum is off Γ.
pub fn e2(table: &VertexTable, long_range: bool, rank_cap: &usize) -> f64 {
    let k0 = table.index_of(Vector3::zeros()).unwrap();
    let eps0 = table.electron_ev[k0];
    let mut sum = 0.0;
    for q in 1..table.len() {
        let neg_q = table.neg(q);
        for nu in 0..table.nmodes {
            let g = if long_range {
                table.full_indices_capped(k0, q, nu, rank_cap)
            } else {
                table.short_indices_capped(k0, q, nu, rank_cap)
            };
            sum += g.norm_sqr()
                / (eps0 - table.electron_ev[neg_q] - table.phonon_ev[q * table.nmodes + nu]);
        }
    }
    sum / table.len() as f64
}

/// The cheap-to-read prefix of a serialized [`VertexTable`]: its identity
/// (grid, band/mode counts, SVD spectrum) without the tab payloads. Field
/// order and types MUST mirror the head of [`VertexTable`] exactly —
/// [`VertexTable::load_head`] deserializes this prefix from the same bincode
/// stream that [`VertexTable::save`] wrote.
#[derive(Debug, Deserialize)]
pub struct VertexTableHead {
    pub grid_n: usize,
    pub nbnd: usize,
    pub nmodes: usize,
    pub pair_offsets: Vec<usize>,
    pub s: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VertexTable {
    pub grid_n: usize,
    pub nbnd: usize,
    pub nmodes: usize,
    pub pair_offsets: Vec<usize>,
    pub s: Vec<f64>,
    /// `[k][component]`, precontracted with the singular value and CBM ket.
    #[serde(alias = "utab")]
    pub wtab: Tab,
    /// CBM bra-band index for each compressed component.
    #[serde(default)]
    pub bra_band: Vec<u8>,
    /// `[point][branch][component]` — flat index `(point*nmodes + branch)*rank + component`.
    pub vtab: Tab,
    /// CBM eigenvector `[k][orbital]`.
    pub cbm_evec: Vec<Cf>,
    /// CBM dispersion in eV, shifted so the Γ band edge is zero.
    pub electron_ev: Vec<f64>,
    /// Physical phonon energies `[q][branch]` in eV.
    pub phonon_ev: Vec<f64>,
    /// Polar Cartesian vertex projected onto phonon branches, `[q][branch]` in eV.
    pub long_ev: Vec<Cf>,
    #[serde(default = "default_phfreq_cutoff_ev")]
    pub phfreq_cutoff_ev: f64,
}

fn default_phfreq_cutoff_ev() -> f64 {
    DEFAULT_PHFREQ_CUTOFF_EV
}

trait StoredComplex: Copy + Send + Sync {
    fn from_f64(value: Cf) -> Self;
}

impl StoredComplex for Complex<f32> {
    fn from_f64(value: Cf) -> Self {
        Self::new(value.re as f32, value.im as f32)
    }
}

impl StoredComplex for Complex64 {
    fn from_f64(value: Cf) -> Self {
        value
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_fourier_tables<T: StoredComplex>(
    epw: &EpwData,
    wigner: &Wigner,
    svd: &EphSvd,
    grid_n: usize,
    point_offset: usize,
    phonon_evec: &[Cf],
    component_bands: &[(usize, usize)],
    wtab: &mut [T],
    vtab: &mut [T],
    completed: &AtomicUsize,
    progress_step: usize,
) {
    let components = svd.rank();
    let nm = epw.nmodes;
    if components == 0 {
        return;
    }
    // The slab may be a shard of the grid; `point_offset` maps the 0-based chunk
    // index to its global grid point (the phase depends on the global point).
    let npoints = wtab.len() / components;
    wtab.par_chunks_mut(components)
        .zip(vtab.par_chunks_mut(nm * components))
        .enumerate()
        .for_each(|(index, (u_row, v_row))| {
            let point = VertexTable::point_for(grid_n, point_offset + index);
            let pe: Vec<Cf> = wigner.k.irvec.iter().map(|&r| phase(point, r)).collect();
            let pp: Vec<Cf> = wigner.g.irvec.iter().map(|&r| phase(point, r)).collect();
            for g in 0..components {
                let (n, m) = component_bands[g];
                let mut u = Cf::new(0.0, 0.0);
                for (re, &p) in pe.iter().enumerate() {
                    let d = wigner.k.ndegen(re, n, m);
                    if d > 0 {
                        u += p * svd.u_at(g, re) / d as f64;
                    }
                }
                u_row[g] = T::from_f64(u);

                for nu in 0..nm {
                    let mut v = Cf::new(0.0, 0.0);
                    for (rp, &p) in pp.iter().enumerate() {
                        for a in 0..nm {
                            let d = wigner.g.ndegen(rp, 0, a / 3);
                            if d > 0 {
                                let e = phonon_evec[(index * nm + nu) * nm + a];
                                v += p * svd.v_at(g, rp, a) * e / d as f64;
                            }
                        }
                    }
                    v_row[nu * components + g] = T::from_f64(v);
                }
            }
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % progress_step == 0 || done == npoints {
                log::info!(
                    "vertex Fourier data: {done}/{npoints} points ({:.0}%)",
                    done * 100 / npoints
                );
            }
        });
}

/// Dispatch the GPU Fourier fill (fills both `wtab` and `vtab` slabs). Split out
/// so the feature gate lives in one place: with a GPU backend it calls the device
/// fill; without one it is a clear runtime error (checked again early in the CLI
/// so a misconfigured job fails before the dataset load and SVD).
#[cfg(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda"))]
#[allow(clippy::too_many_arguments)]
fn gpu_fill_dispatch(
    wigner: &Wigner,
    svd: &EphSvd,
    grid_n: usize,
    lo: usize,
    hi: usize,
    phonon_evec: &[Cf],
    component_bands: &[(usize, usize)],
    chunk_points: usize,
    wtab: &mut [Complex<f32>],
    vtab: &mut [Complex<f32>],
) -> Result<()> {
    crate::gpu::table_fill::fill_fourier_gpu_f32(
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
    Ok(())
}

#[cfg(not(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda")))]
#[allow(clippy::too_many_arguments)]
fn gpu_fill_dispatch(
    _wigner: &Wigner,
    _svd: &EphSvd,
    _grid_n: usize,
    _lo: usize,
    _hi: usize,
    _phonon_evec: &[Cf],
    _component_bands: &[(usize, usize)],
    _chunk_points: usize,
    _wtab: &mut [Complex<f32>],
    _vtab: &mut [Complex<f32>],
) -> Result<()> {
    Err("GPU fill needs a GPU runtime: build with --features gpu-cuda (or gpu-cpu)".into())
}

impl VertexTable {
    pub fn cache_filename(grid_n: usize, total_rank: usize, precision: TablePrecision) -> String {
        format!("table-n{grid_n}-r{total_rank}-{:?}.rmcvtab", precision).to_lowercase()
    }

    pub fn estimated_bytes(
        grid_n: usize,
        components: usize,
        nbnd: usize,
        nmodes: usize,
        precision: TablePrecision,
    ) -> u64 {
        let nk = (grid_n as u64).saturating_pow(3);
        Self::estimated_bytes_for_points(nk, components, nbnd, nmodes, precision)
    }

    /// Peak table bytes for a slab of `points` grid points (a shard fills only
    /// its slab, so the guard scales with the slab, not the full grid).
    fn estimated_bytes_for_points(
        points: u64,
        components: usize,
        nbnd: usize,
        nmodes: usize,
        precision: TablePrecision,
    ) -> u64 {
        let tables = points
            .saturating_mul(components as u64)
            .saturating_mul((1 + nmodes) as u64)
            .saturating_mul(precision.bytes());
        let complex = points
            .saturating_mul((nbnd + nmodes) as u64)
            .saturating_mul(16);
        let real = points
            .saturating_mul((1 + nmodes) as u64)
            .saturating_mul(8)
            .saturating_add((components as u64).saturating_mul(8));
        tables.saturating_add(complex).saturating_add(real)
    }

    /// Full-grid table: the `S=1` case of [`Self::build_range`] with the global
    /// band-edge shift applied.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        epw: &EpwData,
        crystal: &Crystal,
        wigner: &Wigner,
        svd: &EphSvd,
        grid_n: usize,
        phfreq_cutoff_ev: f64,
        precision: TablePrecision,
        max_bytes: u64,
    ) -> Result<Self> {
        let nk = grid_n
            .checked_pow(3)
            .ok_or("grid_n cubed overflows usize")?;
        Self::build_range(
            epw,
            crystal,
            wigner,
            svd,
            grid_n,
            0,
            nk,
            phfreq_cutoff_ev,
            precision,
            max_bytes,
            true,
            FillDevice::Cpu,
        )
    }

    /// Build the table for the point slab `[lo, hi)` of the `n³` grid — the shard
    /// primitive. Every per-point value (base data, Fourier fill, precontraction)
    /// is independent of the slab, so a shard's arrays are byte-for-byte the full
    /// build's corresponding rows. The ONE exception is the band-edge shift, which
    /// is global: pass `apply_shift = false` for shards and let the merge subtract
    /// the global minimum once. `apply_shift = true` (full build) subtracts here.
    #[allow(clippy::too_many_arguments)]
    pub fn build_range(
        epw: &EpwData,
        crystal: &Crystal,
        wigner: &Wigner,
        svd: &EphSvd,
        grid_n: usize,
        lo: usize,
        hi: usize,
        phfreq_cutoff_ev: f64,
        precision: TablePrecision,
        max_bytes: u64,
        apply_shift: bool,
        device: FillDevice,
    ) -> Result<Self> {
        if grid_n == 0 {
            return Err("grid_n must be positive".into());
        }
        if matches!(device, FillDevice::Gpu { .. }) && precision != TablePrecision::F32 {
            return Err("GPU fill produces f32 tables only; use --precision f32".into());
        }
        if !phfreq_cutoff_ev.is_finite() || phfreq_cutoff_ev < 0.0 {
            return Err("phfreq_cutoff_ev must be finite and non-negative".into());
        }
        if (svd.nbnd, svd.nrr_k, svd.nmodes, svd.nrr_g)
            != (epw.nbnd, epw.nrr_k, epw.nmodes, epw.nrr_g)
        {
            return Err("SVD and EPW dimensions differ".into());
        }
        if epw.nbnd > u8::MAX as usize + 1 {
            return Err("precontracted vertex supports at most 256 bands".into());
        }
        let nk = grid_n
            .checked_pow(3)
            .ok_or("grid_n cubed overflows usize")?;
        if lo > hi || hi > nk {
            return Err(format!("shard slab [{lo}, {hi}) is out of range for {nk} points").into());
        }
        let npoints = hi - lo;
        let bytes = Self::estimated_bytes_for_points(
            npoints as u64,
            svd.rank(),
            epw.nbnd,
            epw.nmodes,
            precision,
        );
        log::info!(
            "vertex table estimate: {:.2} GiB ({npoints}/{nk} points)",
            bytes as f64 / 2_f64.powi(30)
        );
        if bytes > max_bytes {
            return Err(format!(
                "USER CHECKPOINT: vertex tables need {:.2} GiB, above the {:.2} GiB limit",
                bytes as f64 / 2_f64.powi(30),
                max_bytes as f64 / 2_f64.powi(30),
            )
            .into());
        }

        let nb = epw.nbnd;
        let nm = epw.nmodes;
        let components = svd.rank();
        let mut electron_ev = vec![0.0; npoints];
        let mut cbm_evec = vec![Cf::new(0.0, 0.0); npoints * nb];
        let mut phonon_ev = vec![0.0; npoints * nm];
        let mut phonon_evec = vec![Cf::new(0.0, 0.0); npoints * nm * nm];
        let mut long_ev = vec![Cf::new(0.0, 0.0); npoints * nm];

        let started = Instant::now();
        let completed = AtomicUsize::new(0);
        let progress_step = (npoints / 20).max(1);
        electron_ev
            .par_iter_mut()
            .zip(cbm_evec.par_chunks_mut(nb))
            .zip(phonon_ev.par_chunks_mut(nm))
            .zip(phonon_evec.par_chunks_mut(nm * nm))
            .zip(long_ev.par_chunks_mut(nm))
            .enumerate()
            .for_each(
                |(index, ((((electron, cbm), phonon), phonon_vectors), long))| {
                    let point = Self::point_for(grid_n, lo + index);
                    let (energies, evecs) = interpolate_h(epw, wigner, point);
                    *electron = energies[0];
                    cbm.copy_from_slice(&evecs[0]);

                    let phonons = interpolate_d_polar(epw, crystal, wigner, point);
                    let gl = g_long_cart(crystal, epw, point);
                    let branches =
                        project_cartesian_branches(crystal, &phonons, &gl, phfreq_cutoff_ev);
                    let evec = quantized_eigenvectors(crystal, &phonons, phfreq_cutoff_ev);
                    for nu in 0..nm {
                        phonon[nu] = phonons.omega_thz[nu].abs() / RY_TO_THZ * RY_TO_EV;
                        phonon_vectors[nu * nm..(nu + 1) * nm]
                            .copy_from_slice(&evec[nu * nm..(nu + 1) * nm]);
                        long[nu] = branches[nu];
                    }
                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % progress_step == 0 || done == npoints {
                        log::info!(
                            "vertex base data: {done}/{npoints} points ({:.0}%)",
                            done * 100 / npoints
                        );
                    }
                },
            );
        log::info!(
            "vertex base data complete: {:.0} points/sec",
            npoints as f64 / started.elapsed().as_secs_f64()
        );
        // Shift to the global band minimum so ε ≥ 0 holds for any material;
        // for a Γ-minimum band (LiF) this equals electron_ev[0]. Shards skip this
        // (their slab min ≠ the global min) — the merge applies it once.
        if apply_shift {
            let band_edge = electron_ev.iter().copied().fold(f64::INFINITY, f64::min);
            if band_edge < electron_ev[0] {
                log::warn!(
                    "conduction minimum is {:.6} eV below Γ — the band edge is off-Γ; \
                     dispersion is shifted to the true minimum",
                    electron_ev[0] - band_edge
                );
            }
            electron_ev.iter_mut().for_each(|e| *e -= band_edge);
        }

        let mut wtab = Tab::zeros(components * npoints, precision);
        let mut vtab = Tab::zeros(components * npoints * nm, precision);

        let mut g_to_nm = vec![(0usize, 0usize); components];
        for n in 0..nb {
            for m in 0..nb {
                for g in svd.pair_range(n, m) {
                    g_to_nm[g] = (n, m);
                }
            }
        }

        let started = Instant::now();
        let completed = AtomicUsize::new(0);
        match device {
            FillDevice::Cpu => match (&mut wtab, &mut vtab) {
                (Tab::F32(w), Tab::F32(v)) => fill_fourier_tables(
                    epw,
                    wigner,
                    svd,
                    grid_n,
                    lo,
                    &phonon_evec,
                    &g_to_nm,
                    w,
                    v,
                    &completed,
                    progress_step,
                ),
                (Tab::F64(w), Tab::F64(v)) => fill_fourier_tables(
                    epw,
                    wigner,
                    svd,
                    grid_n,
                    lo,
                    &phonon_evec,
                    &g_to_nm,
                    w,
                    v,
                    &completed,
                    progress_step,
                ),
                _ => unreachable!("table arrays use one precision"),
            },
            FillDevice::Gpu { chunk_points } => match (&mut wtab, &mut vtab) {
                (Tab::F32(w), Tab::F32(v)) => {
                    gpu_fill_dispatch(
                        wigner,
                        svd,
                        grid_n,
                        lo,
                        hi,
                        &phonon_evec,
                        &g_to_nm,
                        chunk_points,
                        w,
                        v,
                    )?;
                }
                _ => unreachable!("GPU fill is guarded to f32 above"),
            },
        }
        log::info!(
            "vertex Fourier data complete: {:.0} points/sec",
            npoints as f64 / started.elapsed().as_secs_f64()
        );

        let mut table = Self {
            grid_n,
            nbnd: nb,
            nmodes: nm,
            pair_offsets: svd.pair_offsets.clone(),
            s: svd.s.clone(),
            wtab,
            vtab,
            cbm_evec,
            electron_ev,
            phonon_ev,
            long_ev,
            phfreq_cutoff_ev,
            bra_band: vec![],
        };
        table.precontract();
        Ok(table)
    }

    pub fn len(&self) -> usize {
        self.grid_n.pow(3)
    }

    pub fn is_empty(&self) -> bool {
        self.grid_n == 0
    }

    pub fn precision(&self) -> TablePrecision {
        debug_assert_eq!(self.wtab.precision(), self.vtab.precision());
        self.wtab.precision()
    }

    pub fn point(&self, index: usize) -> Vector3<f64> {
        Self::point_for(self.grid_n, index)
    }

    pub(crate) fn point_for(n: usize, index: usize) -> Vector3<f64> {
        let i = index / (n * n);
        let j = index / n % n;
        let k = index % n;
        Vector3::new(
            i as f64 / n as f64,
            j as f64 / n as f64,
            k as f64 / n as f64,
        )
    }

    pub fn index_of(&self, point: Vector3<f64>) -> Option<usize> {
        let mut ijk = [0; 3];
        for axis in 0..3 {
            let scaled = point[axis] * self.grid_n as f64;
            let nearest = scaled.round();
            if (scaled - nearest).abs() > 1e-9 {
                return None;
            }
            ijk[axis] = (nearest as i64).rem_euclid(self.grid_n as i64) as usize;
        }
        Some((ijk[0] * self.grid_n + ijk[1]) * self.grid_n + ijk[2])
    }

    fn cbm(&self, k: usize) -> &[Cf] {
        &self.cbm_evec[k * self.nbnd..(k + 1) * self.nbnd]
    }

    pub fn short_indices(&self, k: usize, q: usize, nu: usize) -> Cf {
        self.short_indices_capped(k, q, nu, &usize::MAX)
    }

    /// `−q mod grid`, purely on flat indices (mirror of [`Self::wrap_add`]).
    #[inline]
    pub fn neg(&self, q: usize) -> usize {
        let n = self.grid_n;
        let nn = n * n;
        let i = (n - q / nn % n) % n;
        let j = (n - q / n % n) % n;
        let l = (n - q % n) % n;
        (i * n + j) * n + l
    }

    /// `(k + q) mod grid`, purely on flat indices — the hot path must not
    /// round-trip through fractional coordinates.
    #[inline]
    pub fn wrap_add(&self, k: usize, q: usize) -> usize {
        let n = self.grid_n;
        let nn = n * n;
        let i = (k / nn + q / nn) % n;
        let j = (k / n % n + q / n % n) % n;
        let l = (k % n + q % n) % n;
        (i * n + j) * n + l
    }

    pub fn short_indices_capped(&self, k: usize, q: usize, nu: usize, rank_cap: &usize) -> Cf {
        debug_assert!(k < self.len() && q < self.len() && nu < self.nmodes);
        let kp = self.wrap_add(k, q);
        let ncomp = self.s.len();
        let u_start = k * ncomp;
        let v_start = (q * self.nmodes + nu) * ncomp;
        let bra = self.cbm(kp);
        let mut out = Cf::new(0.0, 0.0);
        if *rank_cap == usize::MAX {
            for g in 0..ncomp {
                out += bra[self.bra_band[g] as usize].conj()
                    * self.wtab.get(u_start + g)
                    * self.vtab.get(v_start + g);
            }
        } else {
            for pair in 0..self.nbnd * self.nbnd {
                let start = self.pair_offsets[pair];
                let end = start
                    .saturating_add(*rank_cap)
                    .min(self.pair_offsets[pair + 1]);
                for g in start..end {
                    out += bra[self.bra_band[g] as usize].conj()
                        * self.wtab.get(u_start + g)
                        * self.vtab.get(v_start + g);
                }
            }
        }
        out * RY_TO_EV
    }

    pub fn full_indices(&self, k: usize, q: usize, nu: usize) -> Cf {
        self.full_indices_capped(k, q, nu, &usize::MAX)
    }

    pub fn full_indices_capped(&self, k: usize, q: usize, nu: usize, rank_cap: &usize) -> Cf {
        let kp = self.wrap_add(k, q);
        let overlap: Cf = self
            .cbm(kp)
            .iter()
            .zip(self.cbm(k))
            .map(|(bra, ket)| bra.conj() * ket)
            .sum();
        self.short_indices_capped(k, q, nu, rank_cap) + overlap * self.long_ev[q * self.nmodes + nu]
    }

    pub fn full(&self, k: Vector3<f64>, q: Vector3<f64>, nu: usize) -> Cf {
        let ki = self
            .index_of(k)
            .expect("electron momentum is not on the vertex grid");
        let qi = self
            .index_of(q)
            .expect("phonon momentum is not on the vertex grid");
        self.full_indices(ki, qi, nu)
    }

    /// Validate compressed values against the direct oracle on random grid triples.
    pub fn max_direct_error(&self, direct: &DirectVertex<'_>, samples: usize, seed: u64) -> f64 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut worst: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for _ in 0..samples {
            let k = rng.random_range(0..self.len());
            let q = rng.random_range(1..self.len());
            let nu = rng.random_range(0..self.nmodes);
            let expected = direct.full(self.point(k), self.point(q))[nu];
            scale = scale.max(expected.norm());
            worst = worst.max((self.full_indices(k, q, nu) - expected).norm());
        }
        worst / scale
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        write_cache_header(&mut writer, self.precision(), false)?;
        bincode::serialize_into(&mut writer, self)?;
        writer.flush()?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let (expected, is_shard) = read_cache_header(&mut reader)?;
        if is_shard {
            return Err("this is a shard file — reassemble the table with `table-merge`".into());
        }
        let table: Self = bincode::deserialize_from(&mut reader)?;
        if table.wtab.precision() != expected || table.vtab.precision() != expected {
            return Err(
                "vertex-table cache precision flag does not match its payload; rebuild the table"
                    .into(),
            );
        }
        Ok(table)
    }

    /// Read only a table's identity — grid and SVD spectrum — without the tab
    /// payloads (a few KiB instead of up to tens of GiB). Bincode is
    /// sequential, so deserializing the [`VertexTableHead`] prefix from the
    /// same stream `save` wrote reads exactly the leading fields.
    pub fn load_head(path: impl AsRef<Path>) -> Result<VertexTableHead> {
        let mut reader = BufReader::new(File::open(path)?);
        let (_, is_shard) = read_cache_header(&mut reader)?;
        if is_shard {
            return Err("this is a shard file — reassemble the table with `table-merge`".into());
        }
        Ok(bincode::deserialize_from(&mut reader)?)
    }

    /// Write a shard: the header (with the shard flag), the shard metadata, then
    /// the slab table (unshifted, precontracted). Written to a temp file and
    /// renamed into place, so a walltime kill mid-write can never leave a
    /// truncated file that the resume check would treat as already built.
    /// `table-merge` reassembles the full `.rmcvtab` from the complete set.
    pub fn save_shard(&self, meta: &ShardMeta, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let tmp = {
            let mut name = path.as_os_str().to_os_string();
            name.push(".tmp");
            std::path::PathBuf::from(name)
        };
        let mut writer = BufWriter::new(File::create(&tmp)?);
        write_cache_header(&mut writer, self.precision(), true)?;
        bincode::serialize_into(&mut writer, meta)?;
        bincode::serialize_into(&mut writer, self)?;
        writer.flush()?;
        drop(writer);
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Read only a shard's metadata (cheap — skips the slab payload), for the
    /// resume check: an existing shard whose header matches is already built.
    pub fn read_shard_meta(path: impl AsRef<Path>) -> Result<ShardMeta> {
        let mut reader = BufReader::new(File::open(path)?);
        let (_, is_shard) = read_cache_header(&mut reader)?;
        if !is_shard {
            return Err("not a shard file (full vertex table?)".into());
        }
        Ok(bincode::deserialize_from(&mut reader)?)
    }

    /// Load a full shard (metadata + slab table).
    pub fn load_shard(path: impl AsRef<Path>) -> Result<(ShardMeta, Self)> {
        let mut reader = BufReader::new(File::open(path)?);
        let (expected, is_shard) = read_cache_header(&mut reader)?;
        if !is_shard {
            return Err("not a shard file (full vertex table?)".into());
        }
        let meta: ShardMeta = bincode::deserialize_from(&mut reader)?;
        let table: Self = bincode::deserialize_from(&mut reader)?;
        if table.wtab.precision() != expected || table.vtab.precision() != expected {
            return Err("shard precision flag does not match its payload; rebuild".into());
        }
        Ok((meta, table))
    }

    /// Reassemble the full table from its shards: validate they are the complete
    /// contiguous balanced partition of one build, concatenate the per-point
    /// arrays, then apply the deferred global band-edge shift once. The result is
    /// byte-for-byte identical to the equivalent full [`Self::build`] (per-point
    /// work is deterministic and slab-independent; `f64::min` is order-invariant).
    pub fn merge_shards(mut shards: Vec<(ShardMeta, Self)>) -> Result<Self> {
        if shards.is_empty() {
            return Err("no shards to merge".into());
        }
        shards.sort_by_key(|(meta, _)| meta.lo);
        let (head_meta, head) = &shards[0];
        let grid_n = head.grid_n;
        let expected_shards = head_meta.shards;
        let nk = grid_n
            .checked_pow(3)
            .ok_or("grid_n cubed overflows usize")?;
        if shards.len() != expected_shards {
            return Err(format!(
                "expected {expected_shards} shards, got {} — refusing a partial set",
                shards.len()
            )
            .into());
        }
        // Reference invariants come from shard 0; every shard must match it and
        // occupy exactly its balanced slab, with no gaps or overlaps.
        let nb = head.nbnd;
        let nm = head.nmodes;
        let pair_offsets = head.pair_offsets.clone();
        let s = head.s.clone();
        let bra_band = head.bra_band.clone();
        let phfreq_cutoff_ev = head.phfreq_cutoff_ev;
        let precision = head.precision();
        let expected_rank = head_meta.total_rank;
        let mut cursor = 0;
        for (i, (meta, table)) in shards.iter().enumerate() {
            let lo = i * nk / expected_shards;
            let hi = (i + 1) * nk / expected_shards;
            if meta.shards != expected_shards
                || meta.shard != i
                || meta.lo != lo
                || meta.hi != hi
                || meta.lo != cursor
            {
                return Err(
                    "shards are not the complete contiguous balanced partition of one build".into(),
                );
            }
            if meta.total_rank != expected_rank || meta.precision != precision {
                return Err("shards disagree on SVD budget/precision; not from one build".into());
            }
            cursor = meta.hi;
            if table.grid_n != grid_n
                || table.nbnd != nb
                || table.nmodes != nm
                || table.pair_offsets != pair_offsets
                || table.s != s
                || table.phfreq_cutoff_ev != phfreq_cutoff_ev
                || table.precision() != precision
            {
                return Err("shards disagree on grid/SVD/precision; not from one build".into());
            }
            if table.electron_ev.len() != hi - lo {
                return Err("shard slab length does not match its header range".into());
            }
        }
        if cursor != nk {
            return Err("shards do not cover the full grid".into());
        }

        let wtab = Tab::concat(shards.iter().map(|(_, t)| &t.wtab), precision)?;
        let vtab = Tab::concat(shards.iter().map(|(_, t)| &t.vtab), precision)?;
        let mut cbm_evec = Vec::with_capacity(nk * nb);
        let mut electron_ev = Vec::with_capacity(nk);
        let mut phonon_ev = Vec::with_capacity(nk * nm);
        let mut long_ev = Vec::with_capacity(nk * nm);
        for (_, table) in &shards {
            cbm_evec.extend_from_slice(&table.cbm_evec);
            electron_ev.extend_from_slice(&table.electron_ev);
            phonon_ev.extend_from_slice(&table.phonon_ev);
            long_ev.extend_from_slice(&table.long_ev);
        }

        // The global shift the shards deferred; identical to the full build's.
        let band_edge = electron_ev.iter().copied().fold(f64::INFINITY, f64::min);
        if band_edge < electron_ev[0] {
            log::warn!(
                "conduction minimum is {:.6} eV below Γ — the band edge is off-Γ; \
                 dispersion is shifted to the true minimum",
                electron_ev[0] - band_edge
            );
        }
        electron_ev.iter_mut().for_each(|e| *e -= band_edge);

        Ok(Self {
            grid_n,
            nbnd: nb,
            nmodes: nm,
            pair_offsets,
            s,
            wtab,
            bra_band,
            vtab,
            cbm_evec,
            electron_ev,
            phonon_ev,
            long_ev,
            phfreq_cutoff_ev,
        })
    }

    fn precontract(&mut self) {
        let ncomp = self.s.len();
        // Per-point work over the stored points — the full grid for a normal
        // table, a slab for a shard (electron_ev is always one entry per point).
        let npoints = self.electron_ev.len();
        self.bra_band = vec![0; ncomp];
        for n in 0..self.nbnd {
            for m in 0..self.nbnd {
                let pair = n * self.nbnd + m;
                for g in self.pair_offsets[pair]..self.pair_offsets[pair + 1] {
                    self.bra_band[g] = n as u8;
                    for k in 0..npoints {
                        let ket = self.cbm_evec[k * self.nbnd + m];
                        self.wtab.scale(k * ncomp + g, self.s[g] * ket);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epw::{EpmatWp, WsGrid};

    #[test]
    fn wrap_add_matches_the_fractional_round_trip() {
        let table = VertexTable {
            grid_n: 5,
            nbnd: 1,
            nmodes: 1,
            pair_offsets: vec![0, 0],
            s: vec![],
            wtab: Tab::F64(vec![]),
            bra_band: vec![],
            vtab: Tab::F64(vec![]),
            cbm_evec: vec![Cf::new(1.0, 0.0); 125],
            electron_ev: vec![0.0; 125],
            phonon_ev: vec![1.0; 125],
            long_ev: vec![Cf::new(0.0, 0.0); 125],
            phfreq_cutoff_ev: DEFAULT_PHFREQ_CUTOFF_EV,
        };
        for k in 0..table.len() {
            let negated = table.index_of(-table.point(k)).unwrap();
            assert_eq!(table.neg(k), negated);
            for q in 0..table.len() {
                let expected = table.index_of(table.point(k) + table.point(q)).unwrap();
                assert_eq!(table.wrap_add(k, q), expected);
            }
        }
    }

    #[test]
    fn exact_periodic_grid_lookup() {
        let table = VertexTable {
            grid_n: 4,
            nbnd: 1,
            nmodes: 1,
            pair_offsets: vec![0, 0],
            s: vec![],
            wtab: Tab::F64(vec![]),
            bra_band: vec![],
            vtab: Tab::F64(vec![]),
            cbm_evec: vec![Cf::new(1.0, 0.0); 64],
            electron_ev: vec![0.0; 64],
            phonon_ev: vec![1.0; 64],
            long_ev: vec![Cf::new(0.0, 0.0); 64],
            phfreq_cutoff_ev: DEFAULT_PHFREQ_CUTOFF_EV,
        };
        assert_eq!(table.index_of(Vector3::new(0.25, -0.25, 1.0)), Some(28));
        assert_eq!(table.point(28), Vector3::new(0.25, 0.75, 0.0));
        assert_eq!(table.index_of(Vector3::new(0.2, 0.0, 0.0)), None);
    }

    #[test]
    fn binary_cache_round_trips_both_precisions() {
        for precision in [TablePrecision::F32, TablePrecision::F64] {
            let table = VertexTable {
                grid_n: 1,
                nbnd: 1,
                nmodes: 1,
                pair_offsets: vec![0, 1],
                s: vec![2.0],
                wtab: match precision {
                    TablePrecision::F32 => Tab::F32(vec![Complex::new(1.0, -2.0)]),
                    TablePrecision::F64 => Tab::F64(vec![Cf::new(1.0, -2.0)]),
                },
                bra_band: vec![0],
                vtab: match precision {
                    TablePrecision::F32 => Tab::F32(vec![Complex::new(3.0, 4.0)]),
                    TablePrecision::F64 => Tab::F64(vec![Cf::new(3.0, 4.0)]),
                },
                cbm_evec: vec![Cf::new(1.0, 0.0)],
                electron_ev: vec![0.25],
                phonon_ev: vec![0.5],
                long_ev: vec![Cf::new(0.75, -0.25)],
                phfreq_cutoff_ev: DEFAULT_PHFREQ_CUTOFF_EV,
            };
            let path = std::env::temp_dir().join(format!(
                "rmc-vertex-table-{}-{precision:?}.rmcvtab",
                std::process::id()
            ));
            table.save(&path).unwrap();
            let loaded = VertexTable::load(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            assert_eq!(loaded, table);
        }
    }

    #[test]
    fn binary_cache_rejects_wrong_version() {
        let path = std::env::temp_dir().join(format!(
            "rmc-vertex-table-version-{}.rmcvtab",
            std::process::id()
        ));
        let mut header = CACHE_MAGIC.to_vec();
        header.extend_from_slice(&(CACHE_VERSION + 1).to_le_bytes());
        header.extend_from_slice(&0_u32.to_le_bytes());
        std::fs::write(&path, header).unwrap();
        let error = VertexTable::load(&path).unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert!(error.to_string().contains("rebuild the table"));
    }

    /// A tiny but complete EPW dataset — interpolates at any grid_n, so the same
    /// fixture drives the oracle, shard, and GPU-parity tests. Deliberately NOT
    /// minimal: two R-vectors on every Wigner grid (point-dependent dispersion,
    /// phonons, and phases — a shard/fill that mixes up global vs. slab-local
    /// indices, drops the deferred band-edge shift, or transposes U_w/V_w
    /// produces different numbers), two atoms (a/3 ∈ {0,1} exercises the per-atom
    /// degeneracy axis), and ndegen entries > 1 (the /d division is live).
    /// `hermitize` runs before every diagonalization, so the raw blocks need not
    /// be Hermitian; imaginary phonon branches are fine (the fill takes |ω|).
    fn synthetic_inputs() -> (EpwData, Crystal, Wigner, EpmatWp) {
        let nm = 6; // 2 atoms × 3
                    // D(R): stiff point-independent diagonal at R=0 plus a small complex
                    // block at R=[1,0,0] so frequencies and eigenvectors vary with q.
        let mut d = vec![Cf::new(0.0, 0.0); nm * nm * 2];
        for a in 0..nm {
            d[(a * nm + a) * 2] = Cf::new(((a + 1) * (a + 1)) as f64, 0.0);
            for b in 0..nm {
                d[(a * nm + b) * 2 + 1] =
                    Cf::new(0.1 * (a + b + 1) as f64, 0.05 * (a as f64 - b as f64));
            }
        }
        let epw = EpwData {
            ef: 0.0,
            nbnd: 2,
            nrr_k: 2,
            nmodes: nm,
            nrr_q: 2,
            nrr_g: 2,
            zstar: vec![[[0.0; 3]; 3]; 2],
            epsi: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            // h[(i·nbnd+j)·nrr_k + r]: R=0 block diag(0, 2) plus a complex
            // R=[1,0,0] block → k-dependent bands with an off-Γ minimum, so the
            // deferred band-edge shift subtracts a nonzero, point-found value.
            h: vec![
                Cf::new(0.0, 0.0),
                Cf::new(0.3, 0.1),
                Cf::new(0.1, 0.0),
                Cf::new(0.05, -0.02),
                Cf::new(0.1, 0.0),
                Cf::new(0.02, 0.04),
                Cf::new(2.0, 0.0),
                Cf::new(-0.2, 0.05),
            ],
            d,
        };
        let crystal = Crystal {
            nat: 2,
            nmodes: nm,
            at: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            bg: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            omega: 100.0,
            alat: 5.0,
            tau: vec![[0.0; 3], [0.25, 0.25, 0.25]],
            ityp: vec![1, 2],
            amass: vec![1.0, 2.0],
        };
        // ndegen > 1 entries sit on the second R-vector of the k and g grids, so
        // both the electron (per band pair) and el-ph (per atom) degeneracy
        // divisions are exercised with d ≠ 1.
        let wigner = Wigner {
            k: WsGrid::from_parts(vec![[0; 3], [1, 0, 0]], 2, 2, vec![1, 1, 1, 1, 1, 2, 2, 1]),
            q: WsGrid::from_parts(vec![[0; 3], [1, 0, 0]], 2, 2, vec![1, 1, 1, 1, 1, 1, 1, 2]),
            g: WsGrid::from_parts(vec![[0; 3], [0, 1, 0]], 2, 2, vec![1, 1, 1, 1, 1, 2, 1, 1]),
        };
        let epmat = EpmatWp {
            nbnd: 2,
            nrr_k: 2,
            nmodes: nm,
            nrr_g: 2,
            data: (0..EpmatWp::expected_len(2, 2, nm, 2))
                .map(|i| Cf::new((i + 1) as f64 * 0.1, ((i % 5) as f64 - 2.0) * 0.1))
                .collect(),
        };
        (epw, crystal, wigner, epmat)
    }

    #[test]
    fn compressed_table_matches_direct_oracle() {
        let (epw, crystal, wigner, epmat) = synthetic_inputs();
        let svd = EphSvd::compress(&epmat, 1e-10, 3);
        assert!(svd.max_relative_error(&epmat) < 1e-9);
        let error = VertexTable::build(
            &epw,
            &crystal,
            &wigner,
            &svd,
            2,
            DEFAULT_PHFREQ_CUTOFF_EV,
            TablePrecision::F64,
            0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("USER CHECKPOINT"));
        let table = VertexTable::build(
            &epw,
            &crystal,
            &wigner,
            &svd,
            2,
            DEFAULT_PHFREQ_CUTOFF_EV,
            TablePrecision::F64,
            DEFAULT_MAX_BYTES,
        )
        .unwrap();
        let direct = DirectVertex::new(&epw, &crystal, &wigner, &epmat);
        assert!(table.max_direct_error(&direct, 200, 19) < 1e-9);
    }

    #[test]
    fn sharded_build_merges_to_the_full_table() {
        let (epw, crystal, wigner, epmat) = synthetic_inputs();
        let svd = EphSvd::compress(&epmat, 1e-10, 3);
        // n=5 → 125 points, not divisible by 4: exercises the exact balanced
        // slab arithmetic (sizes 31,31,31,32), the risky part of sharding.
        let grid_n = 5_usize;
        let nshards = 4;
        let nk = grid_n.pow(3);
        let precision = TablePrecision::F64;

        let full = VertexTable::build(
            &epw,
            &crystal,
            &wigner,
            &svd,
            grid_n,
            DEFAULT_PHFREQ_CUTOFF_EV,
            precision,
            DEFAULT_MAX_BYTES,
        )
        .unwrap();

        let shards: Vec<_> = (0..nshards)
            .map(|i| {
                let lo = i * nk / nshards;
                let hi = (i + 1) * nk / nshards;
                let table = VertexTable::build_range(
                    &epw,
                    &crystal,
                    &wigner,
                    &svd,
                    grid_n,
                    lo,
                    hi,
                    DEFAULT_PHFREQ_CUTOFF_EV,
                    precision,
                    DEFAULT_MAX_BYTES,
                    false,
                    FillDevice::Cpu,
                )
                .unwrap();
                let meta = ShardMeta {
                    grid_n,
                    shards: nshards,
                    shard: i,
                    lo,
                    hi,
                    total_rank: 3,
                    precision,
                };
                (meta, table)
            })
            .collect();

        // Fixture-strength guards: the merge's deferred band-edge shift is only
        // tested if the raw (unshifted) energies are point-dependent with a
        // nonzero minimum — a point-independent fixture makes this test vacuous.
        let raw_min = shards
            .iter()
            .flat_map(|(_, t)| t.electron_ev.iter().copied())
            .fold(f64::INFINITY, f64::min);
        assert!(raw_min != 0.0, "fixture band edge is 0 — shift untested");
        assert!(
            full.electron_ev.iter().any(|&e| e != full.electron_ev[0]),
            "fixture dispersion is flat — slab indexing untested"
        );

        // A shard round-trips through its file format unchanged.
        let path = std::env::temp_dir().join(format!(
            "rmc-shard-{}-{}.rmcvtab.shard",
            std::process::id(),
            1
        ));
        shards[1].1.save_shard(&shards[1].0, &path).unwrap();
        assert_eq!(VertexTable::read_shard_meta(&path).unwrap(), shards[1].0);
        let (meta, reloaded) = VertexTable::load_shard(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(meta, shards[1].0);
        assert_eq!(reloaded, shards[1].1);

        // Shuffled input order must still reassemble the same table (merge sorts).
        let mut shuffled = shards;
        shuffled.reverse();
        let merged = VertexTable::merge_shards(shuffled).unwrap();
        assert_eq!(merged, full);
    }

    #[test]
    fn merge_refuses_incomplete_shard_sets() {
        let (epw, crystal, wigner, epmat) = synthetic_inputs();
        let svd = EphSvd::compress(&epmat, 1e-10, 3);
        let grid_n = 4_usize;
        let nshards = 4;
        let nk = grid_n.pow(3);
        // Only the first three of four shards — must be refused, not silently
        // merged into a table with a hole.
        let shards: Vec<_> = (0..3)
            .map(|i| {
                let lo = i * nk / nshards;
                let hi = (i + 1) * nk / nshards;
                let table = VertexTable::build_range(
                    &epw,
                    &crystal,
                    &wigner,
                    &svd,
                    grid_n,
                    lo,
                    hi,
                    DEFAULT_PHFREQ_CUTOFF_EV,
                    TablePrecision::F64,
                    DEFAULT_MAX_BYTES,
                    false,
                    FillDevice::Cpu,
                )
                .unwrap();
                let meta = ShardMeta {
                    grid_n,
                    shards: nshards,
                    shard: i,
                    lo,
                    hi,
                    total_rank: 3,
                    precision: TablePrecision::F64,
                };
                (meta, table)
            })
            .collect();
        assert!(VertexTable::merge_shards(shards).is_err());
    }

    #[test]
    fn precontracted_evaluation_matches_band_pair_loop() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(23);
        let grid_n = 2_usize;
        let nk = grid_n.pow(3);
        let nbnd = 2;
        let ncomp = 8;
        let mut table = VertexTable {
            grid_n,
            nbnd,
            nmodes: 1,
            pair_offsets: vec![0, 2, 4, 6, 8],
            s: (0..ncomp).map(|_| rng.random_range(0.2..1.2)).collect(),
            wtab: Tab::F64(
                (0..nk * ncomp)
                    .map(|_| Cf::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                    .collect(),
            ),
            bra_band: vec![],
            vtab: Tab::F64(
                (0..nk * ncomp)
                    .map(|_| Cf::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                    .collect(),
            ),
            cbm_evec: (0..nk * nbnd)
                .map(|_| Cf::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                .collect(),
            electron_ev: vec![0.0; nk],
            phonon_ev: vec![1.0; nk],
            long_ev: vec![Cf::new(0.0, 0.0); nk],
            phfreq_cutoff_ev: DEFAULT_PHFREQ_CUTOFF_EV,
        };
        let raw = match table.wtab.clone() {
            Tab::F64(values) => values,
            Tab::F32(_) => unreachable!(),
        };
        let mut cases = Vec::with_capacity(100);
        for _ in 0..100 {
            let k = rng.random_range(0..nk);
            let q = rng.random_range(0..nk);
            let cap = rng.random_range(1..=3);
            let kp = table.index_of(table.point(k) + table.point(q)).unwrap();
            let mut expected = Cf::new(0.0, 0.0);
            for n in 0..nbnd {
                for m in 0..nbnd {
                    let pair = n * nbnd + m;
                    let end = (table.pair_offsets[pair] + cap).min(table.pair_offsets[pair + 1]);
                    for g in table.pair_offsets[pair]..end {
                        expected += table.s[g]
                            * table.cbm(kp)[n].conj()
                            * table.cbm(k)[m]
                            * raw[k * ncomp + g]
                            * table.vtab.get(q * ncomp + g);
                    }
                }
            }
            cases.push((k, q, cap, expected * RY_TO_EV));
        }

        table.precontract();
        for (k, q, cap, expected) in cases {
            assert!((table.short_indices_capped(k, q, 0, &cap) - expected).norm() < 1.0e-12);
        }
    }

    /// GPU/CPU fill parity: the GPU Fourier fill reproduces the CPU fill within
    /// f32 tolerance (the reductions reorder, so bit-identity is not expected).
    /// Runs on whichever cubecl backend is compiled in — the CPU runtime locally,
    /// CUDA on the A100 (`cargo test --features gpu-cuda`). The kernels use no
    /// plane/shared-memory ops, so the two backends agree.
    #[cfg(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda"))]
    #[test]
    fn gpu_fill_matches_cpu_within_f32_tol() {
        let (epw, crystal, wigner, epmat) = synthetic_inputs();
        let svd = EphSvd::compress(&epmat, 1e-10, 3);
        let grid_n = 4_usize;
        let nk = grid_n.pow(3);
        // Force many small chunks (64 points / 7 per launch → 10 launches) so the
        // chunk-boundary offsets are exercised, not just the single-launch path.
        let build = |device| {
            VertexTable::build_range(
                &epw,
                &crystal,
                &wigner,
                &svd,
                grid_n,
                0,
                nk,
                DEFAULT_PHFREQ_CUTOFF_EV,
                TablePrecision::F32,
                DEFAULT_MAX_BYTES,
                true,
                device,
            )
            .unwrap()
        };
        let cpu = build(FillDevice::Cpu);
        let gpu = build(FillDevice::Gpu { chunk_points: 7 });

        let f32_slice = |t: &Tab| -> Vec<Complex<f32>> {
            match t {
                Tab::F32(v) => v.clone(),
                Tab::F64(_) => unreachable!("f32 table"),
            }
        };
        let max_rel = |a: &[Complex<f32>], b: &[Complex<f32>]| -> f64 {
            assert_eq!(a.len(), b.len());
            let mut scale = 0.0f64;
            let mut worst = 0.0f64;
            for (x, y) in a.iter().zip(b) {
                scale = scale.max((x.re as f64).hypot(x.im as f64));
                worst = worst.max(((x.re - y.re) as f64).hypot((x.im - y.im) as f64));
            }
            if scale > 0.0 {
                worst / scale
            } else {
                worst
            }
        };
        let w_err = max_rel(&f32_slice(&cpu.wtab), &f32_slice(&gpu.wtab));
        let v_err = max_rel(&f32_slice(&cpu.vtab), &f32_slice(&gpu.vtab));
        assert!(w_err < 1e-5, "wtab relative error {w_err:e}");
        assert!(v_err < 1e-5, "vtab relative error {v_err:e}");
    }
}
