//! CPU reference for the GPU batched-chain semantics.
//!
//! ponytail: this is a direct executable spec, not a fast CPU engine; split SoA buffers out only
//! when the GPU host path needs the same code.

use rand::Rng;
use rmc_core::mc::{Measurement, SimulationStats, StepOutcome, Update};
use rmc_core::Merge;

use crate::config::RunConfig;
use crate::flat::philox::{keyed_draw, PhiloxRng};
use crate::flat::updates::{default_update_set, FlatPolaronUpdate};
use crate::flat::FlatDiagram;
use crate::measurement::{PolaronMeasurement, PolaronStats};
use crate::physics;
use crate::update_stats::UpdateStatEntry;

#[derive(Clone, Debug)]
pub struct BatchedRunOutput {
    pub stats: SimulationStats,
    pub measurement: PolaronStats,
    pub update_stats: Vec<UpdateStatEntry>,
    /// Final state of every chain, for GPU-vs-CPU parity checks.
    pub chains: Vec<FlatDiagram>,
}

pub fn run_batched(
    cfg: &RunConfig,
    n_chains: usize,
    group_size: usize,
) -> rmc_core::Result<BatchedRunOutput> {
    assert!(n_chains > 0, "batched run needs at least one chain");
    assert!(group_size > 0, "group_size must be > 0");

    let params = cfg.simulation_params();
    params.validate()?;
    let expected_samples = cfg.max_steps.div_ceil(cfg.steps_per_cycle.max(1)) as usize;
    let mut chains = (0..n_chains)
        .map(|_| build_flat_diagram(cfg))
        .collect::<Vec<_>>();
    let mut updates = (0..n_chains)
        .map(|_| default_update_set())
        .collect::<rmc_core::Result<Vec<_>>>()?;
    let mut measurements = chains
        .iter()
        .map(|chain| {
            PolaronMeasurement::new_flat(
                cfg.num_bins,
                cfg.max_tau,
                cfg.n_batches,
                expected_samples,
                cfg.energy_estimate,
                usize::MAX,
                cfg.period_multiplier,
                chain,
            )
        })
        .collect::<Vec<_>>();

    let mut update_stats = update_stat_templates(&updates[0]);
    let mut samples_done = 0usize;
    let mut self_consistent_period = cfg.initial_self_consistent_period;
    let mut stats = SimulationStats::default();
    while stats.steps_done < params.max_steps {
        for _ in 0..params.steps_per_cycle {
            if stats.steps_done >= params.max_steps {
                break;
            }
            step_groups(
                cfg.seed,
                stats.steps_done,
                group_size,
                &mut chains,
                &mut updates,
                &mut update_stats,
            );
            stats.steps_done += 1;
        }
        for (measurement, chain) in measurements.iter_mut().zip(&chains) {
            measurement.measure(chain);
        }
        samples_done += 1;
        maybe_reweight_uniformly(
            cfg,
            samples_done,
            chains[0].momentum_out(),
            &mut measurements,
            &mut self_consistent_period,
        );
        stats.cycles_done += 1;
    }

    let measurement = measurements_into_group_batches(measurements, cfg.n_batches, n_chains)
        .reduce(Merge::merge)
        .expect("n_chains > 0");
    stats.steps_done *= n_chains as u64;
    stats.cycles_done *= n_chains as u64;
    Ok(BatchedRunOutput {
        stats,
        measurement,
        update_stats,
        chains,
    })
}

pub(crate) fn build_flat_diagram(cfg: &RunConfig) -> FlatDiagram {
    FlatDiagram::with_parameters(
        cfg.alpha,
        cfg.mu,
        cfg.momentum,
        cfg.max_tau,
        cfg.start_tau,
        cfg.min_order,
        cfg.max_order,
        cfg.max_order_gpu,
    )
}

fn step_groups(
    seed: u64,
    step: u64,
    group_size: usize,
    chains: &mut [FlatDiagram],
    updates: &mut [rmc_core::mc::WeightedUpdateSet<FlatPolaronUpdate>],
    counters: &mut [UpdateStatEntry],
) {
    for (group_id, (chain_group, update_group)) in chains
        .chunks_mut(group_size)
        .zip(updates.chunks_mut(group_size))
        .enumerate()
    {
        let update_index = group_update_index(seed, group_id as u64, step);
        for (offset, (chain, updates)) in chain_group.iter_mut().zip(update_group).enumerate() {
            let chain_id = (group_id * group_size + offset) as u64;
            let mut rng = PhiloxRng::new(seed, chain_id, step);
            let outcome = step_selected(chain, updates, update_index, &mut rng);
            let counter = &mut counters[outcome.update_index];
            counter.proposed += 1;
            counter.impossible += u64::from(outcome.impossible);
            counter.accepted += u64::from(outcome.accepted);
            counter.acc_ratio = if counter.proposed == 0 {
                0.0
            } else {
                counter.accepted as f64 / counter.proposed as f64
            };
        }
    }
}

fn group_update_index(seed: u64, group_id: u64, step: u64) -> usize {
    (keyed_draw(seed, group_id, step, u32::MAX)[0] as usize) % 8
}

fn step_selected<R: Rng + ?Sized>(
    chain: &mut FlatDiagram,
    updates: &mut rmc_core::mc::WeightedUpdateSet<FlatPolaronUpdate>,
    update_index: usize,
    rng: &mut R,
) -> StepOutcome {
    let entry = &mut updates.entries_mut()[update_index];
    let probability = entry.update_mut().attempt(chain, rng) * entry.ratio();
    let (accepted, impossible) = if probability < 0.0 {
        entry.update_mut().reject(chain);
        (false, true)
    } else if probability >= 1.0 || rng.gen::<f64>() < probability {
        entry.update_mut().accept(chain);
        (true, false)
    } else {
        entry.update_mut().reject(chain);
        (false, false)
    };
    StepOutcome {
        update_index,
        probability,
        accepted,
        impossible,
    }
}

pub(crate) fn update_stat_templates(
    updates: &rmc_core::mc::WeightedUpdateSet<FlatPolaronUpdate>,
) -> Vec<UpdateStatEntry> {
    updates
        .entries()
        .iter()
        .map(|entry| {
            UpdateStatEntry::from_counts(entry.update().name().to_string(), entry.weight(), 0, 0, 0)
        })
        .collect()
}

pub(crate) fn maybe_reweight_uniformly(
    cfg: &RunConfig,
    samples_done: usize,
    momentum_out: crate::physics::Vec3,
    measurements: &mut [PolaronMeasurement],
    self_consistent_period: &mut usize,
) {
    if samples_done <= *self_consistent_period {
        return;
    }
    let next_period = ((*self_consistent_period as f64) * cfg.period_multiplier) as usize;
    if samples_done + next_period > cfg.max_steps.div_ceil(cfg.steps_per_cycle.max(1)) as usize {
        *self_consistent_period = usize::MAX;
        return;
    }

    let Some(window) = measurements
        .iter()
        .map(|measurement| measurement.stats().clone())
        .reduce(Merge::merge)
    else {
        return;
    };
    let estimate = window.jackknife_energy();
    if !estimate.mean.is_finite() {
        return;
    }
    let energy_estimate = physics::bare_dispersion(momentum_out) + estimate.mean;
    *self_consistent_period = next_period;
    for measurement in measurements {
        measurement.reset_flat_energy_window(energy_estimate, next_period);
    }
}

pub(crate) fn measurements_into_group_batches(
    measurements: Vec<PolaronMeasurement>,
    n_batches: usize,
    n_chains: usize,
) -> impl Iterator<Item = PolaronStats> {
    measurements
        .into_iter()
        .enumerate()
        .map(move |(chain, measurement)| {
            let batch = (chain * n_batches / n_chains).min(n_batches - 1);
            measurement
                .finish()
                .into_single_jackknife_batch(batch, n_batches)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_update_selection_is_shared_within_group() {
        let update = group_update_index(7, 0, 12);
        assert_eq!(update, group_update_index(7, 0, 12));
        assert!(update < 8);
    }

    #[test]
    fn batched_run_records_all_chain_samples() {
        let cfg = RunConfig {
            max_steps: 25,
            steps_per_cycle: 5,
            n_batches: 4,
            num_bins: 20,
            chains: 4,
            initial_self_consistent_period: usize::MAX,
            ..RunConfig::default()
        };

        let output = run_batched(&cfg, 4, 2).unwrap();
        assert_eq!(output.stats.steps_done, 100);
        assert_eq!(output.stats.cycles_done, 20);
        assert_eq!(output.measurement.sample_count, 20);
        assert_eq!(output.measurement.zeroth.total_count(), 20);
        assert_eq!(
            output
                .update_stats
                .iter()
                .map(|row| row.proposed)
                .sum::<u64>(),
            100
        );
    }
}
