//! Phase A4/A5 parity: the CubeCL MC kernel against the flat batched driver
//! on whatever runtime the feature set selects — the CPU runtime locally,
//! real CUDA hardware when built with `--features gpu-cuda` (the A5 parity
//! checkpoint). Exact per-update accept counts, final chain states, caches,
//! and reports; `lanes = 1` reproduces the flat reduction order, the wider
//! lane count exercises the cooperative strided dot + shared reduce.

#![cfg(any(feature = "gpu-cpu", feature = "gpu-cuda", feature = "gpu-hip"))]

mod common;

use common::synthetic_model;
use nalgebra::Vector3;
use rmc_ab_initio::flat::batched::{run_batched, BatchedRunOutput};
use rmc_ab_initio::gpu::run::{run_device, DeviceRunOpts};
use rmc_ab_initio::gpu::SelectedRuntime;
use rmc_diagmc::DiagConfig;

const MAX_ORDER_GPU: usize = 24;

fn config(fixed_tau: bool) -> DiagConfig {
    DiagConfig {
        mu: if fixed_tau { -0.5 } else { 1.0 },
        p_ext: Vector3::zeros(),
        max_tau: 8.0,
        num_bins: 20,
        min_order: 0,
        max_order: 20,
        tau_fit: 4.0,
        seed: 4242,
        chains: 8,
        max_steps: 3000,
        warmup_steps: 100,
        steps_per_cycle: 5,
        n_batches: 4,
        fixed_tau,
        end_time_margin_secs: 60.0,
    }
}

fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    if a.is_nan() && b.is_nan() {
        return;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!(
        (a - b).abs() <= tol * scale,
        "{what}: {a} vs {b} (rel {})",
        (a - b).abs() / scale
    );
}

fn compare(cpu: &BatchedRunOutput, gpu: &BatchedRunOutput, tol: f64, label: &str) {
    // Exact accept-count parity per update type.
    for (c, g) in cpu.update_stats.iter().zip(&gpu.update_stats) {
        assert_eq!(c.name, g.name, "{label}: update order");
        assert_eq!(
            (c.proposed, c.accepted, c.impossible),
            (g.proposed, g.accepted, g.impossible),
            "{label}: {} counts",
            c.name
        );
    }
    // Final chain states: integer bookkeeping exact, floats within tol.
    assert_eq!(cpu.chains.len(), gpu.chains.len());
    for (i, (c, g)) in cpu.chains.iter().zip(&gpu.chains).enumerate() {
        assert_eq!(c.len, g.len, "{label}: chain {i} len");
        assert_eq!(c.order, g.order, "{label}: chain {i} order");
        for key in 0..c.len {
            assert_eq!(
                c.p_out[key], g.p_out[key],
                "{label}: chain {i} p_out[{key}]"
            );
            assert_eq!(c.q[key], g.q[key], "{label}: chain {i} q[{key}]");
            assert_eq!(
                c.branch[key], g.branch[key],
                "{label}: chain {i} branch[{key}]"
            );
            assert_eq!(c.link[key], g.link[key], "{label}: chain {i} link[{key}]");
            assert_close(
                c.tau[key],
                g.tau[key],
                tol,
                &format!("{label}: chain {i} tau[{key}]"),
            );
        }
        assert!(g.check_consistency(), "{label}: chain {i} inconsistent");
        assert_close(
            c.log_modulus,
            g.log_modulus,
            tol,
            &format!("{label}: chain {i} log_modulus"),
        );
        assert_close(
            c.phase.re,
            g.phase.re,
            tol,
            &format!("{label}: chain {i} phase.re"),
        );
    }
    // Report scalars.
    let cpu_report = cpu.report.as_ref().unwrap();
    let gpu_report = gpu.report.as_ref().unwrap();
    assert_close(
        cpu_report.e0_thermo.mean,
        gpu_report.e0_thermo.mean,
        tol,
        &format!("{label}: e0_thermo"),
    );
    assert_close(
        cpu_report.mean_order.mean,
        gpu_report.mean_order.mean,
        tol,
        &format!("{label}: mean_order"),
    );
    assert_close(
        cpu_report.sign.mean,
        gpu_report.sign.mean,
        tol,
        &format!("{label}: sign"),
    );
    assert_eq!(cpu_report.steps_done, gpu_report.steps_done);
}

#[test]
fn kernel_matches_flat_driver_on_cpu_runtime() {
    let model = synthetic_model(7);
    for fixed_tau in [true, false] {
        let cfg = config(fixed_tau);
        let cpu = run_batched(&model, &cfg, 0, MAX_ORDER_GPU).unwrap();
        assert!(
            cpu.report.as_ref().unwrap().mean_order.mean > 0.05,
            "walk never left order 0 (fixed_tau {fixed_tau})"
        );
        // 32 lanes (a real warp) on hardware; 4 on the CPU runtime, whose
        // oversubscribed spin barriers make wide cubes pathologically slow.
        let wide = if cfg!(feature = "gpu-cuda") { 32u32 } else { 4 };
        for lanes in [1u32, wide] {
            let client = <SelectedRuntime as cubecl::prelude::Runtime>::client(&Default::default());
            let gpu = run_device(
                client,
                &model,
                &cfg,
                DeviceRunOpts {
                    max_order_gpu: MAX_ORDER_GPU,
                    lanes,
                    ..Default::default()
                },
            )
            .unwrap();
            // The CPU runtime's math lowering (exp/log1p) differs from
            // glibc by ~1e-12 relative on single draws; the plan's acceptance
            // bar is 1e-6, accept counts are exact.
            let tol = 1.0e-9;
            compare(
                &cpu,
                &gpu,
                tol,
                &format!("fixed_tau {fixed_tau} lanes {lanes}"),
            );
        }
    }
}

/// Offset determinism carries to the device path: two half-populations at
/// offsets 0 and 4 give the same final states as chains 0..8 of one run.
#[test]
fn device_chain_offset_matches_full_run() {
    let model = synthetic_model(7);
    let mut cfg = config(true);
    cfg.chains = 4;
    let mut full_cfg = cfg.clone();
    full_cfg.chains = 8;
    let client = <SelectedRuntime as cubecl::prelude::Runtime>::client(&Default::default());
    let opts = || DeviceRunOpts {
        max_order_gpu: MAX_ORDER_GPU,
        lanes: 1,
        ..Default::default()
    };
    let full = run_device(client.clone(), &model, &full_cfg, opts()).unwrap();
    let lo = run_device(client.clone(), &model, &cfg, opts()).unwrap();
    let hi = run_device(
        client,
        &model,
        &cfg,
        DeviceRunOpts {
            chain_offset: 4,
            ..opts()
        },
    )
    .unwrap();
    let split: Vec<_> = lo.chains.iter().chain(&hi.chains).collect();
    for (a, b) in full.chains.iter().zip(split) {
        assert_eq!(a, b);
    }
}

/// The slurm deadline stops the segment loop between launches, keeping every
/// replayed sample (devel-QoS jobs shut down cleanly instead of being killed).
#[test]
fn device_deadline_truncates_between_segments() {
    let model = synthetic_model(7);
    let mut cfg = config(true);
    cfg.max_steps = 2_000_000;
    // No warmup and a generous budget: the first launch pays the JIT, and the
    // deadline must land after at least one measured segment.
    cfg.warmup_steps = 0;
    let client = <SelectedRuntime as cubecl::prelude::Runtime>::client(&Default::default());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let opts = |deadline| DeviceRunOpts {
        max_order_gpu: MAX_ORDER_GPU,
        lanes: 1,
        cycles_per_launch: 8,
        deadline: Some(deadline),
        ..Default::default()
    };
    let out = run_device(client.clone(), &model, &cfg, opts(deadline)).unwrap();
    assert!(out.steps_done > 0, "nothing measured before the deadline");
    assert!(
        out.steps_done < cfg.max_steps * cfg.chains,
        "run was not truncated"
    );
    assert_eq!(out.report.as_ref().unwrap().steps_done, out.steps_done);

    // An expired deadline means a warmup-only leg: no report, but the chain
    // states and cycle count still come back for the checkpoint.
    let expired = std::time::Instant::now();
    let warmup_only = run_device(client, &model, &cfg, opts(expired)).unwrap();
    assert!(warmup_only.report.is_none());
    assert_eq!(warmup_only.steps_done, 0);
}
