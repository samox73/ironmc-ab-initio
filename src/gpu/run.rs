//! Phase A4 driver: resident chain state on device, segmented kernel
//! launches, host-side measurement replay — `flat::batched::run_batched`
//! semantics exactly (same Philox streams, warmup discard, per-cycle
//! `(τ, order, phase, E_loc)` records into the standard measurement pipeline).
//! Returns the same `BatchedRunOutput`, so rank dumps and the `merge`
//! subcommand work unchanged for GPU ranks.

use std::sync::Arc;

use cubecl::prelude::*;
use num_complex::Complex64;
use rmc_core::{Merge, Result, RmcError};
use rmc_diagmc::measure::{DiagMeasurement, summarize};
use rmc_diagmc::{DiagConfig, Model};
use rmc_frohlich::update_stats::UpdateStatEntry;

use crate::flat::FlatDiagram;
use crate::flat::batched::{BatchedRunOutput, DEFAULT_MAX_ORDER_GPU, build_chain};
use crate::gpu::kernel::mc_kernel;
use crate::gpu::vertex_bench::DeviceTables;
use crate::model_abinitio::AbInitioModel;

/// Cycles per kernel launch: bounds the device sample buffer to
/// `chains · CYCLES_PER_LAUNCH · 36 B` and sets the host sync cadence.
pub const CYCLES_PER_LAUNCH: u64 = 256;
/// Production cube width (one warp per chain). CPU-runtime parity uses 1.
pub const DEFAULT_LANES: u32 = 32;

/// Model arrays beyond the vertex tables: long-range planes, energies, and
/// the proposal-sampling arrays (uploaded verbatim so device decisions replay
/// the host floats exactly).
pub struct DeviceModel {
    pub long_re: cubecl::server::Handle,
    pub long_im: cubecl::server::Handle,
    pub electron_ev: cubecl::server::Handle,
    pub phonon_ev: cubecl::server::Handle,
    pub q_cdf: cubecl::server::Handle,
    pub q_weight: cubecl::server::Handle,
    pub q_weight_sum: f64,
    pub ln_nq: f64,
}

impl DeviceModel {
    pub fn upload<R: Runtime>(client: &ComputeClient<R>, model: &AbInitioModel) -> Self {
        let table = &model.table;
        let long_re: Vec<f64> = table.long_ev.iter().map(|z| z.re).collect();
        let long_im: Vec<f64> = table.long_ev.iter().map(|z| z.im).collect();
        let (q_weight, q_cdf, q_weight_sum) = model.q_sampling();
        Self {
            long_re: client.create_from_slice(f64::as_bytes(&long_re)),
            long_im: client.create_from_slice(f64::as_bytes(&long_im)),
            electron_ev: client.create_from_slice(f64::as_bytes(&table.electron_ev)),
            phonon_ev: client.create_from_slice(f64::as_bytes(&table.phonon_ev)),
            q_cdf: client.create_from_slice(f64::as_bytes(q_cdf)),
            q_weight: client.create_from_slice(f64::as_bytes(q_weight)),
            q_weight_sum,
            ln_nq: model.ln_nq(),
        }
    }
}

/// Device handles for one resident population of chains (SoA, chain-major).
pub struct DeviceState<R: Runtime> {
    pub client: ComputeClient<R>,
    pub chains: u32,
    pub capacity: u32,
    template: FlatDiagram,
    tau: cubecl::server::Handle,
    p_out: cubecl::server::Handle,
    qv: cubecl::server::Handle,
    branch: cubecl::server::Handle,
    link: cubecl::server::Handle,
    len: cubecl::server::Handle,
    log_mod: cubecl::server::Handle,
    phase_re: cubecl::server::Handle,
    phase_im: cubecl::server::Handle,
}

impl<R: Runtime> DeviceState<R> {
    pub fn upload(client: ComputeClient<R>, chains: &[FlatDiagram]) -> Self {
        let template = chains[0].clone();
        let capacity = template.capacity();
        let n = chains.len();
        let mut tau = vec![0.0f64; n * capacity];
        let mut p_out = vec![0u32; n * capacity];
        let mut qv = vec![0u32; n * capacity];
        let mut branch = vec![0u32; n * capacity];
        let mut link = vec![crate::flat::NONE; n * capacity];
        let mut len = vec![0u32; n];
        let mut log_mod = vec![0.0f64; n];
        let mut phase_re = vec![0.0f64; n];
        let mut phase_im = vec![0.0f64; n];
        for (c, chain) in chains.iter().enumerate() {
            assert_eq!(chain.capacity(), capacity);
            let base = c * capacity;
            tau[base..base + capacity].copy_from_slice(&chain.tau);
            p_out[base..base + capacity].copy_from_slice(&chain.p_out);
            qv[base..base + capacity].copy_from_slice(&chain.q);
            branch[base..base + capacity].copy_from_slice(&chain.branch);
            link[base..base + capacity].copy_from_slice(&chain.link);
            len[c] = chain.len as u32;
            log_mod[c] = chain.log_modulus;
            phase_re[c] = chain.phase.re;
            phase_im[c] = chain.phase.im;
        }
        Self {
            chains: n as u32,
            capacity: capacity as u32,
            template,
            tau: client.create_from_slice(f64::as_bytes(&tau)),
            p_out: client.create_from_slice(u32::as_bytes(&p_out)),
            qv: client.create_from_slice(u32::as_bytes(&qv)),
            branch: client.create_from_slice(u32::as_bytes(&branch)),
            link: client.create_from_slice(u32::as_bytes(&link)),
            len: client.create_from_slice(u32::as_bytes(&len)),
            log_mod: client.create_from_slice(f64::as_bytes(&log_mod)),
            phase_re: client.create_from_slice(f64::as_bytes(&phase_re)),
            phase_im: client.create_from_slice(f64::as_bytes(&phase_im)),
            client,
        }
    }

    /// Read the full chain state back (final states, parity tests).
    pub fn download(&self) -> Vec<FlatDiagram> {
        let read_f64 = |h: &cubecl::server::Handle| {
            f64::from_bytes(&self.client.read_one_unchecked(h.clone())).to_vec()
        };
        let read_u32 = |h: &cubecl::server::Handle| {
            u32::from_bytes(&self.client.read_one_unchecked(h.clone())).to_vec()
        };
        let tau = read_f64(&self.tau);
        let p_out = read_u32(&self.p_out);
        let qv = read_u32(&self.qv);
        let branch = read_u32(&self.branch);
        let link = read_u32(&self.link);
        let len = read_u32(&self.len);
        let log_mod = read_f64(&self.log_mod);
        let phase_re = read_f64(&self.phase_re);
        let phase_im = read_f64(&self.phase_im);
        let capacity = self.capacity as usize;
        (0..self.chains as usize)
            .map(|c| {
                let base = c * capacity;
                let mut chain = self.template.clone();
                chain.tau.copy_from_slice(&tau[base..base + capacity]);
                chain.p_out.copy_from_slice(&p_out[base..base + capacity]);
                chain.q.copy_from_slice(&qv[base..base + capacity]);
                chain.branch.copy_from_slice(&branch[base..base + capacity]);
                chain.link.copy_from_slice(&link[base..base + capacity]);
                chain.len = len[c] as usize;
                chain.order = (chain.len - 2) / 2;
                chain.log_modulus = log_mod[c];
                chain.phase = Complex64::new(phase_re[c], phase_im[c]);
                chain
            })
            .collect()
    }
}

pub struct SegmentParams {
    pub step0: u64,
    pub n_cycles: u32,
    pub steps_per_cycle: u32,
}

pub struct SegmentOutput {
    pub tau: Vec<f64>,
    pub eloc: Vec<f64>,
    pub phase_re: Vec<f64>,
    pub phase_im: Vec<f64>,
    pub order: Vec<u32>,
    pub proposed: [u64; 4],
    pub accepted: [u64; 4],
    pub impossible: [u64; 4],
}

/// Launch one segment and read back its sample log (blocks until done).
#[allow(clippy::too_many_arguments)]
pub fn launch_segment<R: Runtime>(
    state: &DeviceState<R>,
    tables: &DeviceTables<R>,
    dmodel: &DeviceModel,
    cfg: &DiagConfig,
    chain_offset: u64,
    max_order_eff: u32,
    lanes: u32,
    params: SegmentParams,
) -> SegmentOutput {
    assert!(params.n_cycles > 0);
    let client = &state.client;
    let chains = state.chains;
    let nk = (tables.grid_n as usize).pow(3);
    let rank = tables.rank as usize;
    let n_samples = params.n_cycles as usize * chains as usize;
    let f64_zeros = |n: usize| client.create_from_slice(f64::as_bytes(&vec![0.0f64; n]));
    let samples_tau = f64_zeros(n_samples);
    let samples_eloc = f64_zeros(n_samples);
    let samples_pre = f64_zeros(n_samples);
    let samples_pim = f64_zeros(n_samples);
    let samples_order = client.create_from_slice(u32::as_bytes(&vec![0u32; n_samples]));
    let stats_zeros = vec![0u32; chains as usize * 4];
    let stats_proposed = client.create_from_slice(u32::as_bytes(&stats_zeros));
    let stats_accepted = client.create_from_slice(u32::as_bytes(&stats_zeros));
    let stats_impossible = client.create_from_slice(u32::as_bytes(&stats_zeros));

    let arr = |h: &cubecl::server::Handle, len: usize| -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(h.clone(), len) }
    };
    let state_len = (chains * state.capacity) as usize;
    let (w_re, w_im, v_re, v_im, cbm, bra_band) = tables.raw_handles();
    mc_kernel::launch::<R>(
        client,
        CubeCount::Static(chains, 1, 1),
        CubeDim::new_1d(lanes),
        arr(&state.tau, state_len),
        arr(&state.p_out, state_len),
        arr(&state.qv, state_len),
        arr(&state.branch, state_len),
        arr(&state.link, state_len),
        arr(&state.len, chains as usize),
        arr(&state.log_mod, chains as usize),
        arr(&state.phase_re, chains as usize),
        arr(&state.phase_im, chains as usize),
        arr(w_re, nk * rank),
        arr(w_im, nk * rank),
        arr(v_re, nk * tables.nmodes as usize * rank),
        arr(v_im, nk * tables.nmodes as usize * rank),
        arr(cbm, 2 * nk * tables.nbnd as usize),
        arr(bra_band, rank),
        arr(&dmodel.long_re, nk * tables.nmodes as usize),
        arr(&dmodel.long_im, nk * tables.nmodes as usize),
        arr(&dmodel.electron_ev, nk),
        arr(&dmodel.phonon_ev, nk * tables.nmodes as usize),
        arr(&dmodel.q_cdf, nk),
        arr(&dmodel.q_weight, nk),
        arr(&samples_tau, n_samples),
        arr(&samples_eloc, n_samples),
        arr(&samples_pre, n_samples),
        arr(&samples_pim, n_samples),
        arr(&samples_order, n_samples),
        arr(&stats_proposed, chains as usize * 4),
        arr(&stats_accepted, chains as usize * 4),
        arr(&stats_impossible, chains as usize * 4),
        cfg.seed as u32,
        (cfg.seed >> 32) as u32,
        params.step0,
        chain_offset,
        chains,
        state.capacity,
        params.n_cycles,
        params.steps_per_cycle,
        tables.grid_n,
        tables.rank,
        tables.nmodes,
        tables.nbnd,
        cfg.mu,
        cfg.max_tau,
        dmodel.q_weight_sum,
        dmodel.ln_nq,
        cfg.min_order as u32,
        max_order_eff,
        if cfg.fixed_tau { 3u32 } else { 4u32 },
    );

    let sum4 = |bytes: &[u8]| -> [u64; 4] {
        let counts = u32::from_bytes(bytes);
        let mut out = [0u64; 4];
        for (i, &c) in counts.iter().enumerate() {
            out[i % 4] += u64::from(c);
        }
        out
    };
    SegmentOutput {
        tau: f64::from_bytes(&client.read_one_unchecked(samples_tau)).to_vec(),
        eloc: f64::from_bytes(&client.read_one_unchecked(samples_eloc)).to_vec(),
        phase_re: f64::from_bytes(&client.read_one_unchecked(samples_pre)).to_vec(),
        phase_im: f64::from_bytes(&client.read_one_unchecked(samples_pim)).to_vec(),
        order: u32::from_bytes(&client.read_one_unchecked(samples_order)).to_vec(),
        proposed: sum4(&client.read_one_unchecked(stats_proposed)),
        accepted: sum4(&client.read_one_unchecked(stats_accepted)),
        impossible: sum4(&client.read_one_unchecked(stats_impossible)),
    }
}

/// Knobs for one device run (or one leg of a checkpointed run).
pub struct DeviceRunOpts {
    pub chain_offset: u64,
    pub max_order_gpu: usize,
    pub lanes: u32,
    pub cycles_per_launch: u64,
    /// Stop between segments once past this instant (see `slurm_deadline`).
    pub deadline: Option<std::time::Instant>,
    /// Resume from a checkpoint: (total cycles already done, final chain
    /// states). Counter-addressable RNG makes the resumed trajectory
    /// bit-identical to an uninterrupted run.
    pub resume: Option<(u64, Vec<FlatDiagram>)>,
    /// Cap this leg at N cycles (deterministic legs; the deadline still
    /// applies on top).
    pub leg_cycles: Option<u64>,
}

impl Default for DeviceRunOpts {
    fn default() -> Self {
        Self {
            chain_offset: 0,
            max_order_gpu: DEFAULT_MAX_ORDER_GPU,
            lanes: DEFAULT_LANES,
            cycles_per_launch: CYCLES_PER_LAUNCH,
            deadline: None,
            resume: None,
            leg_cycles: None,
        }
    }
}

/// Run the full MC (or one leg of it) on device `R`. The returned report and dump
/// carry *this leg's* measured statistics; `cycles_done` is the resume point.
pub fn run_device<R: Runtime>(
    client: ComputeClient<R>,
    model: &Arc<AbInitioModel>,
    cfg: &DiagConfig,
    opts: DeviceRunOpts,
) -> Result<BatchedRunOutput> {
    let DeviceRunOpts {
        chain_offset,
        max_order_gpu,
        lanes,
        cycles_per_launch,
        deadline,
        resume,
        leg_cycles,
    } = opts;
    assert!(cycles_per_launch > 0);
    if cfg.chains == 0 {
        return Err(RmcError::InvalidArgument(
            "device runs must pin `chains` explicitly".to_string(),
        ));
    }
    if cfg.max_steps == 0 || cfg.steps_per_cycle == 0 || cfg.n_batches < 1 {
        return Err(RmcError::InvalidArgument("invalid DiagConfig".to_string()));
    }
    if cfg.max_steps % cfg.steps_per_cycle != 0 {
        return Err(RmcError::InvalidArgument(format!(
            "max_steps ({}) must be a multiple of steps_per_cycle ({})",
            cfg.max_steps, cfg.steps_per_cycle
        )));
    }
    let n_chains = cfg.chains as usize;
    let steps_per_cycle = cfg.steps_per_cycle;
    let expected_samples = cfg.max_steps.div_ceil(steps_per_cycle) as usize;
    let warmup_cycles = cfg.warmup_steps.div_ceil(steps_per_cycle);
    let measured_cycles = expected_samples as u64;
    let total_cycles = warmup_cycles + measured_cycles;
    let max_order_eff = cfg.max_order.min(max_order_gpu) as u32;

    let (cycle_start0, initial_chains) = match resume {
        Some((cycles_done, chains)) => {
            if chains.len() != n_chains {
                return Err(RmcError::InvalidArgument(format!(
                    "checkpoint has {} chains, config wants {n_chains}",
                    chains.len()
                )));
            }
            if chains[0].capacity() != 2 * max_order_gpu + 2 {
                return Err(RmcError::InvalidArgument(
                    "checkpoint capacity differs from --max-order-gpu".to_string(),
                ));
            }
            if cycles_done >= total_cycles {
                return Err(RmcError::InvalidState(format!(
                    "run already complete ({cycles_done}/{total_cycles} cycles)"
                )));
            }
            (cycles_done, chains)
        }
        None => {
            let template = build_chain(model, cfg, max_order_gpu)?;
            (0, vec![template; n_chains])
        }
    };
    let t0 = std::time::Instant::now();
    let tables = DeviceTables::upload(client.clone(), &model.table);
    let dmodel = DeviceModel::upload(&client, model);
    let state = DeviceState::upload(client, &initial_chains);
    drop(initial_chains);
    log::info!("gpu: upload {:.3} s", t0.elapsed().as_secs_f64());

    let mut measurements: Vec<DiagMeasurement> = (0..n_chains)
        .map(|_| {
            DiagMeasurement::new(
                model.clone() as Arc<dyn Model>,
                cfg.num_bins,
                cfg.n_batches,
                expected_samples,
                cfg.max_tau,
            )
        })
        .collect();
    let names: &[&str] = if cfg.fixed_tau {
        &["ChangeInternalTau", "AddPhonon", "RemovePhonon"]
    } else {
        &[
            "ChangeTau",
            "ChangeInternalTau",
            "AddPhonon",
            "RemovePhonon",
        ]
    };
    // Kernel stats use the unified 4-slot indices; a 3-update run leaves
    // slot 0 (ChangeTau) empty.
    let slot0 = if cfg.fixed_tau { 1 } else { 0 };
    let mut update_stats: Vec<UpdateStatEntry> = names
        .iter()
        .map(|name| UpdateStatEntry {
            name: name.to_string(),
            weight: 1.0,
            proposed: 0,
            impossible: 0,
            accepted: 0,
            acc_ratio: 0.0,
        })
        .collect();

    let leg_end = match leg_cycles {
        Some(budget) => total_cycles.min(cycle_start0 + budget),
        None => total_cycles,
    };
    let n_segments = (leg_end - cycle_start0).div_ceil(cycles_per_launch);
    log::info!(
        "gpu: cycles {cycle_start0}..{leg_end} of {total_cycles}, {n_segments} segments of \
         ≤ {cycles_per_launch}; the first launch includes the NVRTC JIT"
    );
    let mut step_secs = 0.0;
    let mut replay_secs = 0.0;
    let mut measured_steps_per_chain = 0u64;
    let mut cycle_start = cycle_start0;
    let mut segment_index = 0u64;
    while cycle_start < leg_end {
        if let Some(deadline) = deadline {
            if std::time::Instant::now() >= deadline {
                break;
            }
        }
        let n_cycles = cycles_per_launch.min(leg_end - cycle_start);
        let t = std::time::Instant::now();
        let segment = launch_segment(
            &state,
            &tables,
            &dmodel,
            cfg,
            chain_offset,
            max_order_eff,
            lanes,
            SegmentParams {
                step0: cycle_start * steps_per_cycle,
                n_cycles: n_cycles as u32,
                steps_per_cycle: steps_per_cycle as u32,
            },
        );
        let segment_secs = t.elapsed().as_secs_f64();
        step_secs += segment_secs;
        segment_index += 1;
        log::info!(
            "gpu: segment {segment_index}/{n_segments} ({n_cycles} cycles) in \
             {segment_secs:.2} s — {:.3e} steps/s",
            (n_cycles * steps_per_cycle * n_chains as u64) as f64 / segment_secs
        );

        let t = std::time::Instant::now();
        for cycle in 0..n_cycles as usize {
            if cycle_start + (cycle as u64) < warmup_cycles {
                continue;
            }
            measured_steps_per_chain += steps_per_cycle;
            for (chain, measurement) in measurements.iter_mut().enumerate() {
                let i = cycle * n_chains + chain;
                measurement.measure_sample(
                    segment.tau[i],
                    segment.order[i] as usize,
                    Complex64::new(segment.phase_re[i], segment.phase_im[i]),
                    segment.eloc[i],
                );
            }
        }
        for (entry, slot) in update_stats.iter_mut().zip(slot0..4) {
            entry.proposed += segment.proposed[slot];
            entry.accepted += segment.accepted[slot];
            entry.impossible += segment.impossible[slot];
            entry.acc_ratio = if entry.proposed == 0 {
                0.0
            } else {
                entry.accepted as f64 / entry.proposed as f64
            };
        }
        replay_secs += t.elapsed().as_secs_f64();
        cycle_start += n_cycles;
    }
    if cycle_start < leg_end {
        log::warn!(
            "deadline stop at cycle {cycle_start}/{total_cycles}: {} measurement steps per \
             chain this leg; error bars are honest but larger, and the sample count is \
             wall-clock-dependent",
            measured_steps_per_chain
        );
    } else if leg_end < total_cycles {
        log::info!("leg complete: cycles {cycle_start0}..{leg_end} of {total_cycles}");
    }
    let data = measurements
        .into_iter()
        .map(DiagMeasurement::into_output)
        .reduce(Merge::merge)
        .expect("chains > 0");
    let steps_done = measured_steps_per_chain * n_chains as u64;
    let report = if steps_done == 0 {
        None
    } else {
        let mut report = summarize(
            data.clone(),
            model.electron_energy(cfg.p_ext),
            cfg.mu,
            cfg.max_tau,
            cfg.tau_fit,
            cfg.fixed_tau,
        )?;
        report.steps_done = steps_done;
        Some(report)
    };
    let chains = state.download();
    Ok(BatchedRunOutput {
        report,
        data,
        update_stats,
        chains,
        steps_done,
        cycles_done: cycle_start,
        step_secs,
        replay_secs,
    })
}
