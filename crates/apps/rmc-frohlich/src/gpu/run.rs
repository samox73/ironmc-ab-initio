//! GPU run driver: resident chain state on device, segmented kernel launches, host-side
//! measurement replay with the same uniform reweighting schedule as `flat::batched::run_batched`.

use rmc_core::mc::SimulationStats;
use rmc_core::Merge;

use crate::app::{AppResult, RunOutput};
use crate::config::RunConfig;
use crate::flat::batched::{
    build_flat_diagram, maybe_reweight_uniformly, measurements_into_group_batches,
    update_stat_templates, BatchedRunOutput,
};
use crate::flat::updates::default_update_set;
use crate::measurement::PolaronMeasurement;

/// Cycles per kernel launch: bounds the device sample buffer to
/// `chains * CYCLES_PER_LAUNCH * 20 B` and sets the host sync cadence.
/// ponytail: fixed constant; make it a config knob when real-GPU profiling asks for it.
pub const CYCLES_PER_LAUNCH: u64 = 256;

#[cfg(feature = "gpu-cuda")]
pub type SelectedRuntime = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
pub type SelectedRuntime = cubecl::hip::HipRuntime;
#[cfg(all(
    feature = "gpu-cpu",
    not(feature = "gpu-hip"),
    not(feature = "gpu-cuda")
))]
pub type SelectedRuntime = cubecl::cpu::CpuRuntime;

pub fn run_gpu_from_config(cfg: &RunConfig) -> AppResult<RunOutput> {
    let start = std::time::Instant::now();
    #[cfg(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda"))]
    let output = {
        let client =
            <SelectedRuntime as cubecl::prelude::Runtime>::client(&Default::default());
        run_device::<SelectedRuntime>(cfg, client)?
    };
    #[cfg(not(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda")))]
    let output = crate::gpu::kernel::launch_reference_kernel(cfg)?;
    Ok(RunOutput {
        stats: output.stats,
        measurement: output.measurement,
        final_state: None,
        update_stats: output.update_stats,
        wall_secs: start.elapsed().as_secs_f64(),
    })
}

/// Run the full MC on device `R`, mirroring `run_batched` semantics exactly: same Philox streams,
/// same group-uniform update selection (group = workgroup), same per-cycle sampling, and the same
/// externally driven uniform reweighting schedule.
pub fn run_device<R: cubecl::prelude::Runtime>(
    cfg: &RunConfig,
    client: cubecl::prelude::ComputeClient<R>,
) -> rmc_core::Result<BatchedRunOutput> {
    use crate::gpu::kernel::{launch_segment, DeviceState, SegmentParams};
    use crate::gpu::state::GpuStateBuffers;

    let n_chains = cfg.chains as usize;
    assert!(n_chains > 0, "device run needs at least one chain");
    let params = cfg.simulation_params();
    params.validate()?;
    let steps_per_cycle = cfg.steps_per_cycle.max(1);
    let expected_samples = cfg.max_steps.div_ceil(steps_per_cycle) as usize;
    // Warmup runs on device like any other cycles (rounded up to whole ones, so RNG streams stay
    // step-keyed as usual); the host simply discards their samples, mirroring the CPU production
    // run's warmup phase. `max_steps` counts measured steps only, as on the CPU.
    let warmup_cycles = cfg.warmup_steps.div_ceil(steps_per_cycle);
    let measured_cycles = expected_samples as u64;
    let total_cycles = warmup_cycles + measured_cycles;
    let final_cycle_steps = if measured_cycles == 0 {
        0
    } else {
        cfg.max_steps - (measured_cycles - 1) * steps_per_cycle
    };

    let template = build_flat_diagram(cfg);
    let t = std::time::Instant::now();
    let state = DeviceState::upload(client, &GpuStateBuffers::initial(cfg, n_chains));
    log::info!("gpu: state upload {:.3}s", t.elapsed().as_secs_f64());
    let mut measurements = (0..n_chains)
        .map(|_| {
            PolaronMeasurement::new_flat(
                cfg.num_bins,
                cfg.max_tau,
                cfg.n_batches,
                expected_samples,
                cfg.energy_estimate,
                usize::MAX,
                cfg.period_multiplier,
                &template,
            )
        })
        .collect::<Vec<_>>();
    let mut update_stats = update_stat_templates(&default_update_set()?);
    let mut self_consistent_period = cfg.initial_self_consistent_period;
    let mut samples_done = 0usize;

    let mut launch_secs = 0.0;
    let mut replay_secs = 0.0;
    let mut cycle_start = 0u64;
    while cycle_start < total_cycles {
        let n_cycles = CYCLES_PER_LAUNCH.min(total_cycles - cycle_start);
        let contains_final = cycle_start + n_cycles == total_cycles;
        let t = std::time::Instant::now();
        let segment = launch_segment(
            &state,
            cfg,
            SegmentParams {
                step0: cycle_start * steps_per_cycle,
                n_cycles: n_cycles as u32,
                steps_per_cycle: steps_per_cycle as u32,
                last_cycle_steps: if contains_final {
                    final_cycle_steps as u32
                } else {
                    steps_per_cycle as u32
                },
            },
        );
        launch_secs += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();

        // Replay the sample log in cycle order so reweighting fires at the same sample counts as
        // the CPU driver. Warmup cycles are discarded.
        for cycle in 0..n_cycles as usize {
            if cycle_start + (cycle as u64) < warmup_cycles {
                continue;
            }
            for (chain, measurement) in measurements.iter_mut().enumerate() {
                let i = cycle * n_chains + chain;
                measurement.measure_flat_sample(
                    segment.samples_tau[i],
                    segment.samples_order[i] as usize,
                    segment.samples_exact[i],
                );
            }
            samples_done += 1;
            maybe_reweight_uniformly(
                cfg,
                samples_done,
                [cfg.momentum, 0.0, 0.0],
                &mut measurements,
                &mut self_consistent_period,
            );
        }

        for (u, entry) in update_stats.iter_mut().enumerate() {
            entry.proposed += segment.proposed[u];
            entry.accepted += segment.accepted[u];
            entry.impossible += segment.impossible[u];
            entry.acc_ratio = if entry.proposed == 0 {
                0.0
            } else {
                entry.accepted as f64 / entry.proposed as f64
            };
        }

        replay_secs += t.elapsed().as_secs_f64();
        cycle_start += n_cycles;
    }
    log::info!("gpu: launches (incl. JIT + sample readback) {launch_secs:.3}s, host replay {replay_secs:.3}s");

    let t = std::time::Instant::now();
    let measurement = measurements_into_group_batches(measurements, cfg.n_batches, n_chains)
        .reduce(Merge::merge)
        .expect("n_chains > 0");
    log::info!("gpu: measurement merge {:.3}s", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let final_buffers = state.download();
    let chains = (0..n_chains)
        .map(|chain| final_buffers.to_flat_diagram(chain, cfg))
        .collect();
    log::info!("gpu: state download {:.3}s", t.elapsed().as_secs_f64());
    let stats = SimulationStats {
        steps_done: cfg.max_steps * n_chains as u64,
        cycles_done: measured_cycles * n_chains as u64,
    };
    Ok(BatchedRunOutput {
        stats,
        measurement,
        update_stats,
        chains,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_run_smoke() {
        let cfg = RunConfig {
            chains: 4,
            max_steps: 20,
            steps_per_cycle: 5,
            n_batches: 4,
            num_bins: 20,
            initial_self_consistent_period: usize::MAX,
            ..RunConfig::default()
        };
        let output = run_gpu_from_config(&cfg).unwrap();
        assert_eq!(output.stats.steps_done, 80);
        assert_eq!(output.measurement.sample_count, 16);
        assert!(output.final_state.is_none());
        assert_eq!(
            output
                .update_stats
                .iter()
                .map(|row| row.proposed)
                .sum::<u64>(),
            80
        );
    }
}
