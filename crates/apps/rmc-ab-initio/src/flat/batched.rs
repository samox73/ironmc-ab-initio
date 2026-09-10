//! Phase A3: batched lockstep driver — the CPU reference for the GPU run
//! semantics. Chains step in lockstep with group-uniform update selection
//! keyed `(seed, global_group, step)` and per-chain Philox streams keyed
//! `(seed, global_chain, step)`; global ids include the rank's `chain_offset`
//! so offset-split runs merge bit-identically to one big run (the
//! `multi-node` golden determinism guarantee at GPU granularity). Per cycle,
//! each chain records `(τ, order, phase, E_loc)` into the standard
//! `rmc_diagmc` measurement pipeline; warmup cycles run like any other and
//! their samples are discarded.

use std::sync::Arc;

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
    pub report: Report,
    /// Raw batch data (rank dumps / offline merge).
    pub data: MeasurementData,
    pub update_stats: Vec<UpdateStatEntry>,
    /// Final state of every chain, for GPU-vs-CPU parity checks.
    pub chains: Vec<FlatDiagram>,
    /// Measured MC steps summed over chains.
    pub steps_done: u64,
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
    run_batched_with_deadline(model, cfg, chain_offset, group_size, max_order_gpu, deadline)
}

pub fn run_batched_with_deadline(
    model: &Arc<AbInitioModel>,
    cfg: &DiagConfig,
    chain_offset: u64,
    group_size: usize,
    max_order_gpu: usize,
    deadline: Option<std::time::Instant>,
) -> Result<BatchedRunOutput> {
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
    if cfg.max_steps == 0 || cfg.steps_per_cycle == 0 || cfg.n_batches < 2 {
        return Err(RmcError::InvalidArgument("invalid DiagConfig".to_string()));
    }
    let n_chains = cfg.chains as usize;
    let n_updates = if cfg.fixed_tau { 3 } else { 4 };
    let steps_per_cycle = cfg.steps_per_cycle;
    let expected_samples = cfg.max_steps.div_ceil(steps_per_cycle) as usize;
    let warmup_cycles = cfg.warmup_steps.div_ceil(steps_per_cycle);
    let measured_cycles = expected_samples as u64;
    let total_cycles = warmup_cycles + measured_cycles;
    let final_cycle_steps = cfg.max_steps - (measured_cycles - 1) * steps_per_cycle;

    // Per-chain measurement memory ≈ bins × batches × 32 B must stay small
    // (the Fröhlich sizing lesson); warn loudly before it hurts at GPU scale.
    let per_chain_bytes = cfg.num_bins * cfg.n_batches * 32;
    if per_chain_bytes > 1 << 20 {
        log::warn!(
            "per-chain measurement memory is {per_chain_bytes} B (> 1 MiB); \
             reduce num_bins × n_batches before scaling chains"
        );
    }

    let template = build_chain(model, cfg, max_order_gpu)?;
    let mut chains: Vec<FlatDiagram> = vec![template; n_chains];
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

    let mut step_secs = 0.0;
    let mut replay_secs = 0.0;
    let mut measured_steps_per_chain = 0u64;
    for cycle in 0..total_cycles {
        if let Some(deadline) = deadline {
            if std::time::Instant::now() >= deadline {
                break;
            }
        }
        let measured = cycle >= warmup_cycles;
        let steps = if measured && cycle + 1 == total_cycles {
            final_cycle_steps
        } else {
            steps_per_cycle
        };
        let t0 = std::time::Instant::now();
        for s in 0..steps {
            let step = cycle * steps_per_cycle + s;
            for (local_group, (chain_group, update_group)) in chains
                .chunks_mut(group_size)
                .zip(updates.chunks_mut(group_size))
                .enumerate()
            {
                let group_id = chain_offset / group_size as u64 + local_group as u64;
                let update_index =
                    (keyed_draw(cfg.seed, group_id, step, u32::MAX)[0] as usize) % n_updates;
                for (offset, (chain, updates)) in
                    chain_group.iter_mut().zip(update_group).enumerate()
                {
                    let chain_id = chain_offset + (local_group * group_size + offset) as u64;
                    let mut rng = PhiloxRng::new(cfg.seed, chain_id, step);
                    let (accepted, impossible) =
                        step_selected(chain, updates, update_index, &mut rng);
                    let counter = &mut update_stats[update_index];
                    counter.proposed += 1;
                    counter.impossible += u64::from(impossible);
                    counter.accepted += u64::from(accepted);
                }
            }
        }
        step_secs += t0.elapsed().as_secs_f64();

        if measured {
            measured_steps_per_chain += steps;
            let t0 = std::time::Instant::now();
            for (chain, measurement) in chains.iter().zip(&mut measurements) {
                measurement.measure_sample(
                    chain.tau_ext(),
                    chain.order,
                    chain.phase,
                    weight::local_energy(chain, model),
                );
            }
            replay_secs += t0.elapsed().as_secs_f64();
        }
    }
    if measured_steps_per_chain < cfg.max_steps {
        log::warn!(
            "deadline stop: {}/{} measurement steps per chain ({:.1}%); error bars are \
             honest but larger, and the sample count is wall-clock-dependent",
            measured_steps_per_chain,
            cfg.max_steps,
            100.0 * measured_steps_per_chain as f64 / cfg.max_steps as f64
        );
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
    let mut report = summarize(
        data.clone(),
        model.electron_energy(cfg.p_ext),
        cfg.mu,
        cfg.max_tau,
        cfg.tau_fit,
        cfg.fixed_tau,
    )?;
    let steps_done = measured_steps_per_chain * n_chains as u64;
    report.steps_done = steps_done;
    Ok(BatchedRunOutput {
        report,
        data,
        update_stats,
        chains,
        steps_done,
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
