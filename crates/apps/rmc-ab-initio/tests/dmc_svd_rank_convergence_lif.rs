#[allow(dead_code)]
#[path = "../examples/dmc_nsv_convergence.rs"]
mod dmc_nsv_convergence;

#[test]
#[ignore = "heavy LiF DMC rank-convergence gate; set IRONMC_DMC_STEPS=100000000 for the production run"]
fn production_budget_agrees_with_untruncated_within_tolerance() {
    // Default to a scout-sized budget so an accidental `--ignored` run costs
    // minutes, not days; the full gate is opt-in via the env var.
    let max_steps = std::env::var("IRONMC_DMC_STEPS")
        .ok()
        .and_then(|steps| steps.parse().ok())
        .unwrap_or(2_000_000);
    let Some(results) = dmc_nsv_convergence::convergence(max_steps).unwrap() else {
        return;
    };
    assert_eq!(results.len(), 2);
    assert_eq!((results[0].total_rank, results[1].total_rank), (800, 1600));
    let difference = (results[0].mean - results[1].mean).abs();
    // Pure 2σ can never pass at production statistics for any finite budget;
    // the physical criterion is the ≤3 meV compression bar. History of this
    // gate doing its job (journal): [150,300] failed at 4.6 meV (6³-provenance
    // tensor); [300,1600] failed at 4.6 ± 0.8 meV (8×8×8 tensor, 2026-09-04),
    // which promoted the production budget from 300 to 800.
    let tolerance = (2.0 * results[0].stderr.hypot(results[1].stderr)).max(0.003);
    assert!(
        difference < tolerance,
        "rank-800/1600 difference {difference:.6} eV exceeds {tolerance:.6} eV"
    );
}
