//! Phase A3: batched-driver semantics — the golden determinism test
//! (offset-split runs merge bit-identically to one big run) and statistical
//! agreement with the standard slotmap runner.

mod common;

use std::sync::Arc;

use common::synthetic_model;
use nalgebra::Vector3;
use rmc_ab_initio::dump::{merge_dumps, RankDump};
use rmc_ab_initio::flat::batched::{run_batched, run_batched_with_deadline};
use rmc_ab_initio::model_abinitio::AbInitioModel;
use rmc_diagmc::{DiagConfig, Model};

fn test_config(chains: u64) -> DiagConfig {
    DiagConfig {
        // Above the band minimum: the tail weight grows with τ (bounded by
        // max_tau), so the decay-fit window is well populated. These tests
        // check determinism, not physics.
        mu: 1.0,
        p_ext: Vector3::zeros(),
        max_tau: 8.0,
        num_bins: 20,
        min_order: 0,
        max_order: 20,
        tau_fit: 4.0,
        seed: 99,
        chains,
        max_steps: 4000,
        warmup_steps: 200,
        steps_per_cycle: 5,
        n_batches: 4,
        fixed_tau: false,
        end_time_margin_secs: 60.0,
    }
}

fn dump(model: &AbInitioModel, cfg: &DiagConfig, rank: u64, output: &rmc_ab_initio::flat::batched::BatchedRunOutput) -> RankDump {
    RankDump {
        config: cfg.clone(),
        rank,
        leg: 0,
        chains: cfg.chains,
        electron_energy_pext: model.electron_energy(cfg.p_ext),
        steps_done: output.steps_done,
        data: output.data.clone(),
        code: "test".to_string(),
    }
}

#[test]
fn offset_split_runs_merge_bit_identically() {
    let model = synthetic_model(7);
    let full_cfg = test_config(4);
    let full = run_batched(&model, &full_cfg, 0, 2, 24).unwrap();
    let full_report = full.report.as_ref().unwrap();
    assert!(full_report.mean_order.mean > 0.05, "walk never left order 0");

    let half_cfg = test_config(2);
    let rank0 = run_batched(&model, &half_cfg, 0, 2, 24).unwrap();
    let rank1 = run_batched(&model, &half_cfg, 2, 2, 24).unwrap();

    // Half-run configs differ from the full config only in `chains`; merging
    // needs identical configs across ranks, which these are.
    let dumps = vec![
        dump(&model, &half_cfg, 0, &rank0),
        dump(&model, &half_cfg, 1, &rank1),
    ];
    let merged = merge_dumps(dumps).unwrap();
    // Debug-string equality: bit-identical floats (shortest-roundtrip) while
    // treating the NaN placeholders in empty green_energy bins as equal.
    assert_eq!(format!("{merged:?}"), format!("{full_report:?}"));

    // Input order must not matter (merge sorts by rank).
    let dumps = vec![
        dump(&model, &half_cfg, 1, &rank1),
        dump(&model, &half_cfg, 0, &rank0),
    ];
    assert_eq!(
        format!("{:?}", merge_dumps(dumps).unwrap()),
        format!("{full_report:?}")
    );
}

#[test]
fn merge_rejects_mismatched_configs_and_duplicate_ranks() {
    let model = synthetic_model(7);
    let cfg = test_config(2);
    let output = run_batched(&model, &cfg, 0, 2, 24).unwrap();
    let mut other_cfg = cfg.clone();
    other_cfg.seed = 100;
    let mismatched = vec![
        dump(&model, &cfg, 0, &output),
        dump(&model, &other_cfg, 1, &output),
    ];
    assert!(merge_dumps(mismatched).is_err());
    let duplicated = vec![
        dump(&model, &cfg, 0, &output),
        dump(&model, &cfg, 0, &output),
    ];
    assert!(merge_dumps(duplicated).is_err());
}

#[test]
fn misaligned_offset_is_rejected() {
    let model = synthetic_model(7);
    let cfg = test_config(2);
    assert!(run_batched(&model, &cfg, 3, 2, 24).is_err());
    let mut auto = cfg.clone();
    auto.chains = 0;
    assert!(run_batched(&model, &auto, 0, 2, 24).is_err());
    // Whole cycles only: a short final cycle would break checkpoint extension.
    let mut ragged = cfg;
    ragged.max_steps = 4001;
    assert!(run_batched(&model, &ragged, 0, 2, 24).is_err());
}

/// Statistical agreement with the standard runner on the production-mode
/// (fixed-τ) thermodynamic estimator: independent RNG machineries, same
/// physics — means must agree within combined error bars.
#[test]
fn batched_driver_agrees_with_slotmap_runner() {
    let model = synthetic_model(7);
    let cfg = DiagConfig {
        mu: -0.5,
        p_ext: Vector3::zeros(),
        max_tau: 8.0,
        num_bins: 20,
        min_order: 0,
        max_order: 20,
        tau_fit: 4.0,
        seed: 1234,
        chains: 4,
        max_steps: 200_000,
        warmup_steps: 20_000,
        steps_per_cycle: 10,
        n_batches: 16,
        fixed_tau: true,
        end_time_margin_secs: 60.0,
    };
    let batched = run_batched(&model, &cfg, 0, 4, 24).unwrap();
    let batched_report = batched.report.as_ref().unwrap();
    let slotmap = rmc_diagmc::run(model.clone() as Arc<dyn Model>, cfg).unwrap();

    let difference = (batched_report.e0_thermo.mean - slotmap.e0_thermo.mean).abs();
    let sigma = batched_report
        .e0_thermo
        .stderr
        .hypot(slotmap.e0_thermo.stderr);
    assert!(
        difference < 5.0 * sigma.max(1.0e-12),
        "e0_thermo disagree: batched {} ± {} vs slotmap {} ± {}",
        batched_report.e0_thermo.mean,
        batched_report.e0_thermo.stderr,
        slotmap.e0_thermo.mean,
        slotmap.e0_thermo.stderr
    );

    let order_difference = (batched_report.mean_order.mean - slotmap.mean_order.mean).abs();
    let order_sigma = batched_report
        .mean_order
        .stderr
        .hypot(slotmap.mean_order.stderr);
    assert!(
        order_difference < 5.0 * order_sigma.max(1.0e-12),
        "mean order disagrees: {} vs {}",
        batched_report.mean_order.mean,
        slotmap.mean_order.mean
    );
}

/// The slurm deadline truncates the cycle loop but keeps every sample and
/// reports the actual (wall-clock-dependent) step count.
#[test]
fn deadline_truncates_but_reports_honestly() {
    let model = synthetic_model(7);
    let mut cfg = test_config(4);
    // Far more work than the deadline allows.
    cfg.max_steps = 50_000_000;
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(300);
    let out = run_batched_with_deadline(&model, &cfg, 0, 2, 24, Some(deadline)).unwrap();
    assert!(out.steps_done > 0, "nothing measured before the deadline");
    assert!(
        out.steps_done < cfg.max_steps * cfg.chains,
        "run was not truncated"
    );
    let out_report = out.report.as_ref().unwrap();
    assert_eq!(out_report.steps_done, out.steps_done);
    assert!(out_report.mean_order.mean.is_finite());

    // An already-expired deadline measures nothing: no report to summarize,
    // but chains + cycles_done still come back so the caller can checkpoint
    // warmup progress instead of losing the leg.
    let expired = std::time::Instant::now();
    let warmup_only = run_batched_with_deadline(&model, &cfg, 0, 2, 24, Some(expired)).unwrap();
    assert!(warmup_only.report.is_none());
    assert_eq!(warmup_only.steps_done, 0);
}
