//! Checkpoint/resume for the batched CPU engine — the CPU twin of
//! `tests/gpu_checkpoint.rs`. The RNG is counter-addressable, so a run split
//! into legs is *bit-identical* to an uninterrupted run in its chain
//! trajectories; the legs' merged report reproduces the single run's estimates
//! (batch partitioning differs, so report equality is value-level, not bit).

mod common;

use common::synthetic_model;
use nalgebra::Vector3;
use rmc_ab_initio::dump::{merge_dumps, same_trajectory, RankDump};
use rmc_ab_initio::flat::batched::{run_batched_opts, BatchedRunOpts, BatchedRunOutput};
use rmc_ab_initio::model_abinitio::AbInitioModel;
use rmc_diagmc::{DiagConfig, Model};

const MAX_ORDER_GPU: usize = 24;

fn config() -> DiagConfig {
    DiagConfig {
        mu: -0.5,
        p_ext: Vector3::zeros(),
        max_tau: 8.0,
        num_bins: 20,
        min_order: 0,
        max_order: 20,
        tau_fit: 4.0,
        seed: 977,
        chains: 8,
        max_steps: 3000,
        warmup_steps: 100,
        steps_per_cycle: 5,
        n_batches: 1,
        fixed_tau: true,
        end_time_margin_secs: 60.0,
    }
}

/// group_size = 1 (per-chain selection), no deadline — the production CPU
/// default. Determinism holds for any fixed group_size; 1 keeps offsets trivial.
fn base_opts() -> BatchedRunOpts {
    BatchedRunOpts {
        max_order_gpu: MAX_ORDER_GPU,
        group_size: 1,
        ..Default::default()
    }
}

fn dump(model: &AbInitioModel, cfg: &DiagConfig, leg: u64, output: &BatchedRunOutput) -> RankDump {
    RankDump {
        config: cfg.clone(),
        rank: 0,
        leg,
        chains: cfg.chains,
        electron_energy_pext: model.electron_energy(cfg.p_ext),
        steps_done: output.steps_done,
        data: output.data.clone(),
        code: "test".to_string(),
    }
}

#[test]
fn resumed_legs_match_uninterrupted_run() {
    let model = synthetic_model(7);
    let cfg = config();
    // 20 warmup + 600 measured cycles; split after 300.
    let full = run_batched_opts(&model, &cfg, base_opts()).unwrap();
    assert_eq!(full.cycles_done, 620);

    let leg1 = run_batched_opts(
        &model,
        &cfg,
        BatchedRunOpts {
            leg_cycles: Some(300),
            ..base_opts()
        },
    )
    .unwrap();
    assert_eq!(leg1.cycles_done, 300);
    assert_eq!(leg1.steps_done, (300 - 20) * 5 * cfg.chains);

    let leg2 = run_batched_opts(
        &model,
        &cfg,
        BatchedRunOpts {
            resume: Some((leg1.cycles_done, leg1.chains.clone())),
            ..base_opts()
        },
    )
    .unwrap();
    assert_eq!(leg2.cycles_done, 620);
    assert_eq!(leg1.steps_done + leg2.steps_done, full.steps_done);

    // Bit-identical trajectories: final chain states equal exactly.
    assert_eq!(leg2.chains, full.chains);

    // Merged leg statistics reproduce the single run's estimates (value-level:
    // the batch partitioning differs between one run and two legs).
    let merged = merge_dumps(vec![
        dump(&model, &cfg, 0, &leg1),
        dump(&model, &cfg, 1, &leg2),
    ])
    .unwrap();
    let full_report = full.report.as_ref().unwrap();
    assert_eq!(merged.steps_done, full_report.steps_done);
    let close = |a: f64, b: f64, what: &str| {
        assert!(
            (a - b).abs() <= 1.0e-12 * a.abs().max(b.abs()).max(1.0),
            "{what}: {a} vs {b}"
        );
    };
    close(
        merged.e0_thermo.mean,
        full_report.e0_thermo.mean,
        "e0_thermo",
    );
    close(
        merged.mean_order.mean,
        full_report.mean_order.mean,
        "mean_order",
    );
    close(merged.sign.mean, full_report.sign.mean, "sign");

    // Guards: wrong resume shapes fail loudly.
    assert!(run_batched_opts(
        &model,
        &cfg,
        BatchedRunOpts {
            resume: Some((620, leg2.chains.clone())),
            ..base_opts()
        },
    )
    .is_err());
    let mut fewer = cfg.clone();
    fewer.chains = 4;
    assert!(run_batched_opts(
        &model,
        &fewer,
        BatchedRunOpts {
            resume: Some((300, leg1.chains.clone())),
            ..base_opts()
        },
    )
    .is_err());
}

/// Extending max_steps on a completed run continues bit-identically: a fresh
/// run with the longer schedule equals complete(short) + resumed extension.
#[test]
fn extending_max_steps_matches_fresh_long_run() {
    let model = synthetic_model(7);
    let short = config();
    let mut long = config();
    long.max_steps = 6000;
    assert!(same_trajectory(&short, &long));

    let fresh = run_batched_opts(&model, &long, base_opts()).unwrap();

    let leg1 = run_batched_opts(&model, &short, base_opts()).unwrap();
    assert_eq!(leg1.cycles_done, 620); // complete under the short schedule
    let leg2 = run_batched_opts(
        &model,
        &long,
        BatchedRunOpts {
            resume: Some((leg1.cycles_done, leg1.chains.clone())),
            ..base_opts()
        },
    )
    .unwrap();
    assert_eq!(leg2.cycles_done, 1220);
    assert_eq!(leg2.chains, fresh.chains);
    assert_eq!(leg1.steps_done + leg2.steps_done, fresh.steps_done);

    // Legs with different max_steps merge (same trajectory).
    let merged = merge_dumps(vec![
        dump(&model, &short, 0, &leg1),
        dump(&model, &long, 1, &leg2),
    ])
    .unwrap();
    let fresh_report = fresh.report.as_ref().unwrap();
    assert_eq!(merged.steps_done, fresh_report.steps_done);
    assert!(
        (merged.e0_thermo.mean - fresh_report.e0_thermo.mean).abs() <= 1.0e-12,
        "{} vs {}",
        merged.e0_thermo.mean,
        fresh_report.e0_thermo.mean
    );
}
