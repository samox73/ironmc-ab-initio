//! Phase A3: batched lockstep driver — the CPU reference for the GPU run
//! semantics. Chains step in lockstep with group-uniform update selection
//! keyed `(seed, global_group, step)` and per-chain Philox streams keyed
//! `(seed, global_chain, step)`; global ids include the rank's `chain_offset`
//! so offset-split runs merge bit-identically to one big run (the
//! `multi-node` golden determinism guarantee at GPU granularity). Per cycle,
//! each chain records `(τ, order, phase, E_loc)` into the standard
//! `rmc_diagmc` measurement pipeline; warmup cycles run like any other and
//! their samples are discarded.
//!
//! Group-uniform selection is a device concession (uniform control flow on
//! SIMT hardware), and it is statistically free: the update-type draw is
//! keyed only by `(seed, group, step)` — never by chain state — so each
//! chain's *marginal* trajectory law is exactly the standard random-scan
//! chain driven by the uniform mixture kernel (1/n)·Σᵤ Pᵤ, identical to
//! per-chain selection, and the 1/n selection probabilities of paired moves
//! (AddPhonon ↔ RemovePhonon) cancel in the acceptance ratio unchanged.
//! Sharing the draw alters only the *joint* law across a group: chains
//! attempting the same move class simultaneously are weakly correlated,
//! which can only inflate variance — a linear chain-average stays unbiased
//! under any cross-chain correlation. The load-bearing invariant is the
//! state-independence of the selection: resampling impossible moves, or any
//! state-dependent group/update choice, would break the 1/n cancellation
//! and genuinely bias the run (see [`step_selected`]).

use std::sync::Arc;

use rayon::prelude::*;
use rmc_core::{Merge, Result, RmcError};
use rmc_diagmc::measure::{summarize, DiagMeasurement, MeasurementData};
use rmc_diagmc::{DiagConfig, Model, Report};
use rmc_frohlich::flat::philox::{keyed_draw, PhiloxRng};
use rmc_frohlich::update_stats::UpdateStatEntry;

use super::updates::{update_set, update_set_fixed_tau, FlatAbUpdate};
use super::{weight, FlatDiagram};
use crate::model_abinitio::AbInitioModel;

pub const DEFAULT_GROUP_SIZE: usize = 64;
pub const DEFAULT_MAX_ORDER_GPU: usize = 256;

pub struct BatchedRunOutput {
    /// `None` for a warmup-only leg (deadline hit before the first measured
    /// cycle): no statistics to summarize, but `chains`/`cycles_done` still
    /// carry the warmup progress for the checkpoint.
    pub report: Option<Report>,
    /// Raw batch data (rank dumps / offline merge).
    pub data: MeasurementData,
    pub update_stats: Vec<UpdateStatEntry>,
    /// Final state of every chain, for GPU-vs-CPU parity checks.
    pub chains: Vec<FlatDiagram>,
    /// Measured MC steps summed over chains.
    pub steps_done: u64,
    /// Total cycles completed (warmup + measured) — the resume point for
    /// checkpointed device runs.
    pub cycles_done: u64,
    /// Wall time inside the step loop only — the kernel-equivalent phase
    /// timer quoted as steps/s.
    pub step_secs: f64,
    pub replay_secs: f64,
}

/// Run with the `SLURM_JOB_END_TIME` deadline armed (minus
/// `cfg.end_time_margin_secs`), like the slotmap runner: the cycle loop stops
/// early, keeps every sample measured so far, and reports an honest
/// (truncated) `steps_done`.
pub fn run_batched(
    model: &Arc<AbInitioModel>,
    cfg: &DiagConfig,
    chain_offset: u64,
    group_size: usize,
    max_order_gpu: usize,
) -> Result<BatchedRunOutput> {
    let deadline = rmc_diagmc::run::slurm_deadline(cfg.end_time_margin_secs)?;
    run_batched_opts(
        model,
        cfg,
        BatchedRunOpts {
            chain_offset,
            group_size,
            max_order_gpu,
            deadline,
            resume: None,
            leg_cycles: None,
        },
    )
}

/// As [`run_batched`] but with an explicit, already-computed deadline.
pub fn run_batched_with_deadline(
    model: &Arc<AbInitioModel>,
    cfg: &DiagConfig,
    chain_offset: u64,
    group_size: usize,
    max_order_gpu: usize,
    deadline: Option<std::time::Instant>,
) -> Result<BatchedRunOutput> {
    run_batched_opts(
        model,
        cfg,
        BatchedRunOpts {
            chain_offset,
            group_size,
            max_order_gpu,
            deadline,
            resume: None,
            leg_cycles: None,
        },
    )
}

/// Knobs for one batched CPU run (or one leg of a checkpointed run) — the CPU
/// twin of [`crate::gpu::run::DeviceRunOpts`].
pub struct BatchedRunOpts {
    pub chain_offset: u64,
    /// Update-selection group size (the per-step draw is keyed by
    /// `chain_id / group_size`). Trajectory-shaping, yet deliberately NOT a
    /// `DiagConfig` field: the config is engine-agnostic (the slotmap engine
    /// has no groups) and bincode-embedded in every dump/checkpoint on disk,
    /// so adding a field there orphans them all. The price is that neither
    /// `same_trajectory` nor the checkpoint can catch a group change — every
    /// leg of a run must repeat the same value. TODO: fold into `DiagConfig`
    /// at the next `FORMAT_VERSION` bump (see the note at its definition).
    pub group_size: usize,
    pub max_order_gpu: usize,
    /// Stop between cycles once past this instant (see `slurm_deadline`).
    pub deadline: Option<std::time::Instant>,
    /// Resume from a checkpoint: (total cycles already done, final chain
    /// states). Counter-addressable RNG makes the resumed trajectory
    /// bit-identical to an uninterrupted run.
    pub resume: Option<(u64, Vec<FlatDiagram>)>,
    /// Cap this leg at N cycles (deterministic legs; the deadline still
    /// applies on top).
    pub leg_cycles: Option<u64>,
}

impl Default for BatchedRunOpts {
    fn default() -> Self {
        Self {
            chain_offset: 0,
            group_size: DEFAULT_GROUP_SIZE,
            max_order_gpu: DEFAULT_MAX_ORDER_GPU,
            deadline: None,
            resume: None,
            leg_cycles: None,
        }
    }
}

/// The batched driver with the full option set: rank offset, deadline,
/// checkpoint resume, and per-leg cycle caps. Chains step in lockstep groups
/// (group-uniform update selection) but never interact, so rayon runs the
/// groups in parallel — the counter-addressable RNG and the fixed global-chain
/// merge order keep the result bit-identical to a single-threaded run.
pub fn run_batched_opts(
    model: &Arc<AbInitioModel>,
    cfg: &DiagConfig,
    opts: BatchedRunOpts,
) -> Result<BatchedRunOutput> {
    let BatchedRunOpts {
        chain_offset,
        group_size,
        max_order_gpu,
        deadline,
        resume,
        leg_cycles,
    } = opts;
    if cfg.chains == 0 {
        return Err(RmcError::InvalidArgument(
            "batched runs must pin `chains` explicitly (auto-sizing would break \
             chain-id offsets)"
                .to_string(),
        ));
    }
    if group_size == 0 || chain_offset % group_size as u64 != 0 {
        return Err(RmcError::InvalidArgument(format!(
            "chain_offset ({chain_offset}) must be a multiple of group_size ({group_size}) \
             so update-selection groups align across ranks"
        )));
    }
    // n_batches = 1 is legal here (per-chain single batch): with many chains
    // the jackknife still has plenty of replicates, and summarize refuses
    // fewer than two non-empty batches overall.
    if cfg.max_steps == 0 || cfg.steps_per_cycle == 0 || cfg.n_batches < 1 {
        return Err(RmcError::InvalidArgument("invalid DiagConfig".to_string()));
    }
    // Whole cycles only: a short final cycle would skip step indices, breaking
    // checkpoint-extension determinism, and buys nothing.
    if cfg.max_steps % cfg.steps_per_cycle != 0 {
        return Err(RmcError::InvalidArgument(format!(
            "max_steps ({}) must be a multiple of steps_per_cycle ({})",
            cfg.max_steps, cfg.steps_per_cycle
        )));
    }
    let n_chains = cfg.chains as usize;
    let n_updates = if cfg.fixed_tau { 3 } else { 4 };
    let steps_per_cycle = cfg.steps_per_cycle;
    let expected_samples = cfg.max_steps.div_ceil(steps_per_cycle) as usize;
    let warmup_cycles = cfg.warmup_steps.div_ceil(steps_per_cycle);
    let measured_cycles = expected_samples as u64;
    let total_cycles = warmup_cycles + measured_cycles;

    // Per-chain measurement memory ≈ bins × batches × 32 B must stay small
    // (the Fröhlich sizing lesson); warn loudly before it hurts at GPU scale.
    let per_chain_bytes = cfg.num_bins * cfg.n_batches * 32;
    if per_chain_bytes > 1 << 20 {
        log::warn!(
            "per-chain measurement memory is {per_chain_bytes} B (> 1 MiB); \
             reduce num_bins × n_batches before scaling chains"
        );
    }

    // Fresh run builds one template per chain; a resumed leg continues from the
    // checkpoint's final chain states (validated against the config).
    let (cycle_start0, mut chains): (u64, Vec<FlatDiagram>) = match resume {
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
    let mut updates = (0..n_chains)
        .map(|_| {
            if cfg.fixed_tau {
                update_set_fixed_tau(model.clone())
            } else {
                update_set(model.clone())
            }
        })
        .collect::<Result<Vec<_>>>()?;
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
    let mut update_stats: Vec<UpdateStatEntry> = updates[0]
        .entries()
        .iter()
        .map(|entry| UpdateStatEntry {
            name: entry.update().name().to_string(),
            weight: entry.weight(),
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
    // Parallelize over CHAINS, each running a whole CHUNK of cycles per rayon
    // task (stepping + measuring fused) — coarse-grained, like the slotmap
    // runner. The natural GPU shape (fork-join per cycle) is a disaster on CPU:
    // it pays a rayon barrier every cycle over tiny tasks and bounces each
    // chain's (multi-KB) state between cores, so ~all wall time goes to overhead
    // (measured: ~36 ms/cycle of which the MC work is µs). Chains never interact,
    // so this is bit-identical; the deadline is checked between chunks, so all
    // chains stay in lockstep at chunk boundaries and `cycles_done` is a single
    // consistent resume point.
    const CYCLES_PER_CHUNK: u64 = 256;
    let compute_start = std::time::Instant::now();
    let mut cycles_done = cycle_start0;
    while cycles_done < leg_end {
        if let Some(deadline) = deadline {
            if std::time::Instant::now() >= deadline {
                break;
            }
        }
        let chunk_lo = cycles_done;
        let chunk_hi = leg_end.min(chunk_lo + CYCLES_PER_CHUNK);
        let chunk_stats: Vec<Vec<(u64, u64, u64)>> = chains
            .par_iter_mut()
            .zip(updates.par_iter_mut())
            .zip(measurements.par_iter_mut())
            .enumerate()
            .map(|(local_chain, ((chain, chain_updates), measurement))| {
                let chain_id = chain_offset + local_chain as u64;
                // Group-uniform update selection is keyed by group id; each chain
                // recomputes its own group's draw (identical within a group), so
                // per-chain parallelism reproduces any group_size exactly.
                let group_id = chain_id / group_size as u64;
                let mut local = vec![(0u64, 0u64, 0u64); n_updates];
                for cycle in chunk_lo..chunk_hi {
                    for s in 0..steps_per_cycle {
                        let step = cycle * steps_per_cycle + s;
                        let update_index =
                            (keyed_draw(cfg.seed, group_id, step, u32::MAX)[0] as usize) % n_updates;
                        let mut rng = PhiloxRng::new(cfg.seed, chain_id, step);
                        let (accepted, impossible) =
                            step_selected(chain, chain_updates, update_index, &mut rng);
                        let counter = &mut local[update_index];
                        counter.0 += 1;
                        counter.1 += u64::from(impossible);
                        counter.2 += u64::from(accepted);
                    }
                    if cycle >= warmup_cycles {
                        measurement.measure_sample(
                            chain.tau_ext(),
                            chain.order,
                            chain.phase,
                            weight::local_energy(chain, model),
                        );
                    }
                }
                local
            })
            .collect();
        for counters in &chunk_stats {
            for (index, (proposed, impossible, accepted)) in counters.iter().enumerate() {
                update_stats[index].proposed += proposed;
                update_stats[index].impossible += impossible;
                update_stats[index].accepted += accepted;
            }
        }
        cycles_done = chunk_hi;
    }
    // Stepping and measurement are fused into one parallel pass, so the whole
    // compute phase is one timer (the old per-cycle step/replay split only made
    // sense when they were separate rayon passes).
    let step_secs = compute_start.elapsed().as_secs_f64();
    let replay_secs = 0.0;
    // Measured steps THIS leg = cycles in [cycle_start0, cycles_done) past warmup.
    let measured_steps_per_chain =
        cycles_done.saturating_sub(cycle_start0.max(warmup_cycles)) * steps_per_cycle;
    if cycles_done < leg_end {
        log::warn!(
            "deadline stop at cycle {cycles_done}/{total_cycles}: {measured_steps_per_chain} \
             measurement steps per chain this leg; error bars are honest but larger, and the \
             sample count is wall-clock-dependent"
        );
    } else if leg_end < total_cycles {
        log::info!("leg complete: cycles {cycle_start0}..{leg_end} of {total_cycles}");
    }
    for counter in &mut update_stats {
        counter.acc_ratio = if counter.proposed == 0 {
            0.0
        } else {
            counter.accepted as f64 / counter.proposed as f64
        };
    }

    // Merge per-chain batch lists in global chain order — the fixed order
    // that makes offset-split runs bit-identical to one big run.
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
    Ok(BatchedRunOutput {
        report,
        data,
        update_stats,
        chains,
        steps_done,
        cycles_done,
        step_secs,
        replay_secs,
    })
}

/// The initial per-chain state (identical for every chain): tail at
/// `max_tau` in fixed-τ mode, `max_tau / 2` otherwise (mirroring
/// `Diagram::new` + the runner's fixed-τ pin), cache synced.
pub fn build_chain(
    model: &AbInitioModel,
    cfg: &DiagConfig,
    max_order_gpu: usize,
) -> Result<FlatDiagram> {
    let table = &model.table;
    let p_ext = table
        .index_of(cfg.p_ext)
        .ok_or_else(|| RmcError::InvalidArgument("p_ext is not on the table grid".to_string()))?
        as u32;
    let start_tau = if cfg.fixed_tau { cfg.max_tau } else { cfg.max_tau / 2.0 };
    let mut chain = FlatDiagram::new(
        table.grid_n as u32,
        cfg.mu,
        p_ext,
        cfg.max_tau,
        start_tau,
        cfg.min_order,
        cfg.max_order,
        max_order_gpu,
    );
    weight::sync_cache(&mut chain, model);
    Ok(chain)
}

/// Attempt one group-selected update; returns `(accepted, impossible)`. An
/// impossible proposal (negative probability) counts as proposed-and-rejected
/// and must NOT be redrawn: resampling until a possible move would make the
/// effective update-type distribution state-dependent, breaking the uniform
/// 1/n selection-probability cancellation in the acceptance ratio — a real
/// bias, not a determinism nit.
fn step_selected<R: rand::Rng + ?Sized>(
    chain: &mut FlatDiagram,
    updates: &mut rmc_core::mc::WeightedUpdateSet<FlatAbUpdate>,
    update_index: usize,
    rng: &mut R,
) -> (bool, bool) {
    use rmc_core::mc::Update;
    let entry = &mut updates.entries_mut()[update_index];
    let probability = entry.update_mut().attempt(chain, rng) * entry.ratio();
    if probability < 0.0 {
        entry.update_mut().reject(chain);
        (false, true)
    } else if probability >= 1.0 || rng.gen::<f64>() < probability {
        entry.update_mut().accept(chain);
        (true, false)
    } else {
        entry.update_mut().reject(chain);
        (false, false)
    }
}
