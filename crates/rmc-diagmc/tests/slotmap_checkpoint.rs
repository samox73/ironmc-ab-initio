//! Multi-node (chain-offset) + checkpoint/resume for the slotmap engine — the
//! CPU-fast engine behind `run-cpu`. Mirrors the batched engine's
//! `rmc-ab-initio/tests/{flat_batched,cpu_checkpoint}.rs`:
//!   * offset-split runs merge bit-identically to one big run (packing
//!     invariance across ranks);
//!   * a run split into legs resumes bit-identically (final `Diagram`s equal).
//!
//! Configs are kept short enough that the drift-check resync never fires
//! (interval = steps_per_cycle * 4096 = 8192 steps), so the cached weight stays
//! purely incremental and the whole `Diagram` (which includes it) is
//! bit-comparable via `PartialEq`.

use std::sync::Arc;

use nalgebra::Vector3;
use rmc_core::Merge;
use rmc_diagmc::{run_leg, summarize, DiagConfig, FrohlichModel, Model};

fn config(chains: u64, max_steps: u64) -> DiagConfig {
    DiagConfig {
        mu: -1.2,
        max_tau: 8.0,
        tau_fit: 4.0,
        num_bins: 32,
        chains,
        max_steps,
        warmup_steps: 200,
        steps_per_cycle: 2,
        n_batches: 4,
        ..DiagConfig::default()
    }
}

#[test]
fn offset_split_runs_merge_bit_identically() {
    let model: Arc<dyn Model> = Arc::new(FrohlichModel::new(1.0));
    let full = run_leg(model.clone(), config(4, 4000), 0, None).unwrap();
    assert!(full.report.mean_order.mean > 0.05, "walk never left order 0");

    let rank0 = run_leg(model.clone(), config(2, 4000), 0, None).unwrap();
    let rank1 = run_leg(model.clone(), config(2, 4000), 2, None).unwrap();

    // Concatenate per-chain batch lists in global chain order (rank 0 then rank
    // 1) — the fixed order that makes an offset-split identical to one big run.
    let mut merged = summarize(
        rank0.data.merge(rank1.data),
        model.electron_energy(Vector3::zeros()),
        -1.2,
        8.0,
        4.0,
        false,
    )
    .unwrap();
    // `summarize` leaves steps_done at 0; `merge_dumps` sums it across ranks.
    merged.steps_done = rank0.steps_done + rank1.steps_done;
    // Debug-string equality: bit-identical floats (shortest-roundtrip), treating
    // NaN placeholders in empty green bins as equal.
    assert_eq!(format!("{merged:?}"), format!("{:?}", full.report));
}

#[test]
fn resumed_legs_match_uninterrupted_run() {
    let model: Arc<dyn Model> = Arc::new(FrohlichModel::new(1.0));
    let full = run_leg(model.clone(), config(4, 4000), 0, None).unwrap();

    let leg1 = run_leg(model.clone(), config(4, 2000), 0, None).unwrap();
    assert!(leg1.complete);
    let leg2 = run_leg(model.clone(), config(4, 2000), 0, Some(leg1.chains.clone())).unwrap();
    assert_eq!(leg1.steps_done + leg2.steps_done, full.steps_done);

    // Bit-identical trajectory: every chain's final Diagram (vertices, momenta,
    // AND cached weight — no resync at this size) equals the uninterrupted run's.
    for (i, ((leg2_d, _), (full_d, _))) in leg2.chains.iter().zip(&full.chains).enumerate() {
        assert_eq!(leg2_d, full_d, "chain {i} diverged on resume");
    }

    // Merged leg statistics reproduce the single run's estimates (value-level:
    // the batch partitioning differs — 8 batches across two legs vs 4).
    let merged = summarize(
        leg1.data.merge(leg2.data),
        model.electron_energy(Vector3::zeros()),
        -1.2,
        8.0,
        4.0,
        false,
    )
    .unwrap();
    let close = |a: f64, b: f64, what: &str| {
        assert!(
            (a - b).abs() <= 1.0e-12 * a.abs().max(b.abs()).max(1.0),
            "{what}: {a} vs {b}"
        );
    };
    close(merged.e0_thermo.mean, full.report.e0_thermo.mean, "e0_thermo");
    close(merged.mean_order.mean, full.report.mean_order.mean, "mean_order");
    close(merged.sign.mean, full.report.sign.mean, "sign");

    // Guards: a resume whose chain count differs from the config fails loudly,
    // and a multi-rank (offset) run must pin `chains` (no `chains = 0`).
    let mismatched = config(2, 2000);
    assert!(run_leg(model.clone(), mismatched, 0, Some(leg1.chains.clone())).is_err());
    let mut auto = config(0, 2000);
    auto.chains = 0;
    assert!(run_leg(model, auto, 2, None).is_err());
}
