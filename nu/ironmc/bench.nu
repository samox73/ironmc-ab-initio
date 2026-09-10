# Benchmarks (run these yourself; agents must not — see AGENTS.md).

use common.nu [ROOT]

# Hot-path vertex evaluation throughput.
export def vertex [] {
    cd $ROOT
    cargo run --release --example vertex_bench -p rmc-ab-initio
}

# Engine steps/sec on the Fröhlich model (Plan C7 fixture).
export def dmc [] {
    cd $ROOT
    cargo run --release --example dmc_bench -p rmc-diagmc
}
