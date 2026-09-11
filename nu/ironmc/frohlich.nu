# Fröhlich polaron targets. Top-level namespaces (ladder, tables, gpu, …)
# are the ab-initio LiF work; everything Fröhlich-specific lives here.

use common.nu [ROOT]

# Engine steps/sec on the Fröhlich model (Plan C7 fixture). Run benches
# yourself; agents must not — see AGENTS.md.
export def bench [] {
    cd $ROOT
    cargo run --release --example dmc_bench -p rmc-diagmc
}
