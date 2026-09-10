//! GPU kernel vs. flat-CPU batched driver parity, executed on the CubeCL CPU runtime.
//!
//! Both sides consume the identical Philox word streams, so trajectories must agree exactly up to
//! transcendental-function ULP differences between Rust's libm and the device runtime. Integer
//! state (topology, storage, counters) is compared exactly; f64 state with a tight tolerance.
//! Runs on whichever runtime feature is enabled (gpu-cpu locally, gpu-hip/gpu-cuda on real cards).
#![cfg(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda"))]

use cubecl::prelude::Runtime;
use rand::Rng;
use rmc_frohlich::config::RunConfig;
use rmc_frohlich::flat::philox::{keyed_draw, PhiloxRng};
use rmc_frohlich::flat::FlatDiagram;
use rmc_frohlich::gpu::kernel::launch_reference_kernel;
use rmc_frohlich::gpu::run::{run_device, SelectedRuntime};

/// The device `u01` mirrors `rand`'s `gen::<f64>()`: words 2i, 2i+1 assembled low-first,
/// `(v >> 11) * 2^-53`. If `rand` ever changes its conversion, this fails.
#[test]
fn rand_f64_conversion_matches_word_stream() {
    let (seed, chain, step) = (0x5150_1234, 7, 42);
    let mut rng = PhiloxRng::new(seed, chain, step);
    for draw in 0u32..64 {
        let word = |w: u32| keyed_draw(seed, chain, step, w / 4)[w as usize % 4];
        let lo = word(2 * draw);
        let hi = word(2 * draw + 1);
        let bits = u64::from(lo) | (u64::from(hi) << 32);
        let expected = (bits >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
        let actual: f64 = rng.gen();
        assert_eq!(actual, expected, "draw {draw}");
    }
}

fn assert_f64_close(label: &str, actual: f64, expected: f64) {
    // ULP-level libm-vs-MLIR differences compound over thousands of steps to ~1e-8 relative;
    // a real logic bug produces O(1) divergence (and exact-count mismatches) long before this.
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= 1e-6 * scale,
        "{label}: actual {actual}, expected {expected}"
    );
}

fn assert_chain_matches(chain: usize, gpu: &FlatDiagram, cpu: &FlatDiagram) {
    assert_eq!(gpu.order, cpu.order, "chain {chain} order");
    assert_eq!(gpu.head, cpu.head, "chain {chain} head");
    assert_eq!(gpu.tail, cpu.tail, "chain {chain} tail");
    assert_eq!(gpu.link, cpu.link, "chain {chain} link");
    assert_eq!(gpu.prev, cpu.prev, "chain {chain} prev");
    assert_eq!(gpu.next, cpu.next, "chain {chain} next");
    assert_eq!(gpu.storage, cpu.storage, "chain {chain} storage");
    assert_eq!(gpu.storage_idx, cpu.storage_idx, "chain {chain} storage_idx");
    assert_eq!(
        gpu.phonons_above, cpu.phonons_above,
        "chain {chain} phonons_above"
    );
    for slot in 0..cpu.capacity() {
        assert_f64_close(
            &format!("chain {chain} tau[{slot}]"),
            gpu.tau[slot],
            cpu.tau[slot],
        );
        for c in 0..3 {
            assert_f64_close(
                &format!("chain {chain} p_out[{slot}][{c}]"),
                gpu.p_out[slot][c],
                cpu.p_out[slot][c],
            );
            assert_f64_close(
                &format!("chain {chain} q[{slot}][{c}]"),
                gpu.q[slot][c],
                cpu.q[slot][c],
            );
        }
    }
}

/// Full-run parity: 130 chains (3 workgroups, one partial), 2001 steps (segmented launches with a
/// partial final cycle), self-consistent reweighting on.
#[test]
fn device_kernel_matches_flat_cpu_driver() {
    let cfg = RunConfig {
        alpha: 2.0,
        chains: 130,
        max_steps: 2001,
        steps_per_cycle: 5,
        n_batches: 8,
        num_bins: 50,
        max_order_gpu: 64,
        initial_self_consistent_period: 50,
        period_multiplier: 2.0,
        seed: 0xB0BA_CAFE_D00D_1234,
        ..RunConfig::default()
    };

    let cpu = launch_reference_kernel(&cfg).unwrap();
    let client = <SelectedRuntime as Runtime>::client(&Default::default());
    let gpu = run_device::<SelectedRuntime>(&cfg, client).unwrap();

    assert_eq!(gpu.stats, cpu.stats);

    assert_eq!(gpu.update_stats.len(), cpu.update_stats.len());
    for (g, c) in gpu.update_stats.iter().zip(&cpu.update_stats) {
        assert_eq!(g.name, c.name);
        assert_eq!(g.proposed, c.proposed, "{} proposed", g.name);
        assert_eq!(g.accepted, c.accepted, "{} accepted", g.name);
        assert_eq!(g.impossible, c.impossible, "{} impossible", g.name);
    }

    assert_eq!(gpu.measurement.sample_count, cpu.measurement.sample_count);
    assert_eq!(
        gpu.measurement.zeroth.total_count(),
        cpu.measurement.zeroth.total_count()
    );
    assert_eq!(
        gpu.measurement.energy_estimates.len(),
        cpu.measurement.energy_estimates.len(),
        "reweighting schedule diverged"
    );
    let g_energy = gpu.measurement.jackknife_energy();
    let c_energy = cpu.measurement.jackknife_energy();
    if c_energy.mean.is_finite() {
        assert_f64_close("jackknife energy", g_energy.mean, c_energy.mean);
    }

    assert_eq!(gpu.chains.len(), cpu.chains.len());
    for (i, (g, c)) in gpu.chains.iter().zip(&cpu.chains).enumerate() {
        assert_chain_matches(i, g, c);
    }
}
