# GPU (CubeCL/CUDA) targets from the gpu-abinitio plan.
#
#   ironmc gpu vbench 20                        # A1 microbench on this node's GPU
#   ironmc gpu vbench 50 --variants coop        # winner-only, biggest grid
#   ironmc slurm submit { ironmc gpu vbench 50 } --partition zen3_0512_a100x2 --devel --out-dir runs/gpu/vbench
#
# Each target does `cuda load` itself, so it works interactively and inside
# sbatch scripts alike (the toolkit must exist: `ironmc cuda setup`, once per
# machine — $HOME is shared with the compute nodes).

use common.nu [ROOT table-path config-names run-dirs]
use cuda.nu

# A1 vertex-dot microbenchmark (GO/NO-GO): prints the node's cpu-host rayon
# baseline plus per-variant device evals/s and table-stream GB/s, with the
# 1e-6 correctness gate asserted first.
export def vbench [
    grid: int = 20                # table grid n → datasets/LiF/table-n{n}-r{rank}-f32.rmcvtab
    --rank: int = 800
    --evals: int = 4194304
    --seed: int = 7
    --variants: string = "scalar,vector,coop"
] {
    cuda load
    cd $ROOT
    let table = (table-path $grid $rank)
    if not ($table | path exists) {
        error make {msg: $"no table at ($table); build it with `ironmc tables`"}
    }
    (cargo run --release -p rmc-ab-initio --features gpu-cuda --
        vbench --table $table --evals $evals --seed $seed --variants $variants)
}

# A4/A5 device run of a DiagConfig on this node's GPU; writes a RankDump
# for `rmc-ab-initio merge`. `config` may be a path or a name from
# crates/apps/rmc-ab-initio/configs (e.g. lif-gpu-tau60).
export def run [
    grid: int = 20
    --config: string@config-names = "lif-gpu"
    --rank: int                   # chain-offset rank; defaults to $SLURM_PROCID (multi-rank srun)
    --chains: int = 0             # override the config's chain count (0 = keep config)
    --rank-table: int = 800       # table rank in the artifact filename
    --lanes: int = 32
    --max-order-gpu: int = 256
    --cycles-per-launch: int = 256
    --leg-cycles: int = 0         # cap this leg at N cycles (0 = run to completion/deadline)
    --p-ext: list<float>          # external momentum in fractional coords, e.g. [0.5 0.0 0.5] = X;
                                  # must be a grid point. Overrides the config's p_ext and becomes
                                  # part of the run's config — resubmit legs with the same value
    --out-dir: string@run-dirs    # run directory (e.g. runs/gpu/n50-prod): holds dumps +
                                  # checkpoints; resubmitting with the same dir resumes the run
                                  # (bit-identical to an uninterrupted one). Defaults to
                                  # $IRONMC_OUT_DIR (set by `slurm submit --out-dir`); REQUIRED otherwise.
] {
    let out_dir = ($out_dir | default $env.IRONMC_OUT_DIR?)
    if $out_dir == null {
        error make {msg: "every run must be named: pass --out-dir runs/gpu/<run-name> (or submit via `slurm submit --out-dir`)"}
    }
    cuda load
    cd $ROOT
    # Code runs from $ROOT (a pinned worktree under `slurm submit
    # --commit-hash`); data — tables, run dirs — always lives in the main
    # repo, which pinned jobs pass via $IRONMC_DATA_ROOT.
    let data_root = ($env.IRONMC_DATA_ROOT? | default ($ROOT | path expand))
    let rank = ($rank | default ($env.SLURM_PROCID? | default 0 | into int))
    let table = ($data_root | path join datasets LiF $"table-n($grid)-r($rank_table)-f32.rmcvtab")
    let out_dir = if ($out_dir | str starts-with "/") {
        $out_dir
    } else {
        $data_root | path join $out_dir
    }
    let config_path = if ($config | path exists) {
        $config
    } else {
        $ROOT | path join crates apps rmc-ab-initio configs $"($config).json"
    }
    (cargo run --release -p rmc-ab-initio --features gpu-cuda --
        run-gpu --table $table --config $config_path --rank $rank --chains $chains
        --p-ext ($p_ext | default [] | str join ",")
        --lanes $lanes --max-order-gpu $max_order_gpu
        --cycles-per-launch $cycles_per_launch --leg-cycles $leg_cycles --out-dir $out_dir)
}

# Chain-count scaling probe: short runs at increasing populations with
# per-segment logs — localizes where throughput falls off a cliff. Runs
# largest-last so a hang still leaves the smaller points in the log.
export def scale [
    grid: int = 20
    --chains-list: list<int> = [256, 1024, 4096, 8192]
    --config: string@config-names = "lif-gpu-smoke"
    --cycles-per-launch: int = 8
] {
    cuda load
    cd $ROOT
    let table = (table-path $grid 800)
    let config_path = ($ROOT | path join crates apps rmc-ab-initio configs $"($config).json")
    for chains in $chains_list {
        print $"=== chains ($chains) ==="
        (cargo run --release -p rmc-ab-initio --features gpu-cuda --
            run-gpu --table $table --config $config_path --chains $chains
            --cycles-per-launch $cycles_per_launch --out-dir $"runs/gpu-scale/($chains)")
    }
}

# Merge every RankDump (all ranks, all legs) in a results dir into one
# report. This is the cumulative statistics of a checkpointed run; the
# per-job result-*.json files are per-leg views, not merge inputs.
export def merge [
    dir: string@run-dirs          # e.g. runs/gpu/n50-prod
    --out: string                 # merged report path (default <dir>/merged.json)
] {
    cd $ROOT
    let dumps = (glob ($dir | path join "batches-rank*.bin"))
    if ($dumps | is-empty) {
        error make {msg: $"no batches-rank*.bin in ($dir)"}
    }
    let out = ($out | default ($dir | path join "merged.json"))
    (^target/release/rmc-ab-initio merge --out $out ...$dumps)
}

# A5 CUDA parity checkpoint: the engine parity suite on this node's GPU.
export def parity [] {
    cuda load
    cd $ROOT
    cargo test --release -p rmc-ab-initio --features gpu-cuda --test gpu_engine_parity -- --nocapture
}

# Vertex-table GPU-fill parity on this node's GPU: the Fourier fill (`--device gpu`)
# vs the CPU fill, per-element within f32 tolerance. A lib unit test, so `--lib`
# (the engine parity above is an integration test → `--test`).
export def table-parity [] {
    cuda load
    cd $ROOT
    (cargo test --release -p rmc-ab-initio --features gpu-cuda --lib
        gpu_fill_matches_cpu_within_f32_tol -- --nocapture)
}

# nvidia-smi snapshot of the visible devices (sanity check inside jobs).
export def info [] {
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
}
