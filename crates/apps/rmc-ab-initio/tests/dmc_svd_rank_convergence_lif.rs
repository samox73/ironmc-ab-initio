#[allow(dead_code)]
#[path = "../examples/dmc_nsv_convergence.rs"]
mod dmc_nsv_convergence;

#[test]
#[ignore = "100M-step LiF DMC rank-convergence gate"]
fn total_ranks_150_and_300_agree_within_two_combined_sigma() {
    let Some(results) = dmc_nsv_convergence::convergence(100_000_000).unwrap() else {
        return;
    };
    assert_eq!(results.len(), 2);
    assert_eq!((results[0].total_rank, results[1].total_rank), (150, 300));
    let difference = (results[0].mean - results[1].mean).abs();
    let two_sigma = 2.0 * results[0].stderr.hypot(results[1].stderr);
    assert!(
        difference < two_sigma,
        "rank-150/300 difference {difference:.6} eV exceeds 2σ={two_sigma:.6} eV"
    );
}
