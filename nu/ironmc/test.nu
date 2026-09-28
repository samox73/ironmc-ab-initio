# Test tiers. Heavy #[ignore] gates live in gates.nu, not here.

use common.nu [ROOT]

# The whole workspace (fast + dataset-gated suites, no ignored tests).
export def all [] {
    cd $ROOT
    cargo test --workspace
}

# The LiF dataset ladder: readers → physics oracles → full-tensor round-trip
# → compression convergence. Run after any dataset regeneration.
export def lif [] {
    cd $ROOT
    cargo test --release -p rmc-ab-initio --test datasets_lif --test reconstruct_lif --test frohlich_vertex_lif --test epb_lif --test model_lif --test vertex_direct_lif --test vertex_table_lif --test svd_rank_convergence_lif
}
