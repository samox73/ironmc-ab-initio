# Quantum ESPRESSO / EPW dataset regeneration (LiF).

use common.nu [DATASET]

# The full scf → ph → nscf → epw pipeline. Interactive: it pauses at the
# manual window-check before EPW (see the Wannier-window notes in epw.in).
export def pipeline [] {
    cd $DATASET
    nu run_pipeline.nu
}
