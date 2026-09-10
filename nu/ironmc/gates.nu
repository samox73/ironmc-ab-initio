# Convergence gates (decided thresholds; see journal.md for their calibration).

use common.nu [ROOT]

# E² compression gate: global budgets vs the untruncated reference, 8³ (~1 min).
export def e2 [] {
    cd $ROOT
    cargo gate-e2
}

# DMC rank gate [800 vs 1600] at 8³. Scout-sized by default; pass
# --steps 100000000 for the production certification.
export def dmc [--steps: int] {
    cd $ROOT
    if $steps == null {
        cargo gate-dmc
    } else {
        with-env {IRONMC_DMC_STEPS: ($steps | into string)} { cargo gate-dmc }
    }
}
