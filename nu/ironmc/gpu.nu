# GPU (CubeCL/CUDA) targets from the gpu-abinitio plan.
#
#   ironmc gpu vbench 20                        # A1 microbench on this node's GPU
#   ironmc gpu vbench 50 --variants coop        # winner-only, biggest grid
#   ironmc slurm submit gpu vbench 50 --partition zen3_0512_a100x2 --devel
#   (quote the target when it carries flags:
#    ironmc slurm submit "gpu vbench 50 --variants coop" --partition zen3_0512_a100x2)
#
# Each target does `cuda load` itself, so it works interactively and inside
# sbatch scripts alike (the toolkit must exist: `ironmc cuda setup`, once per
# machine — $HOME is shared with the compute nodes).

use common.nu [ROOT table-path]
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
# crates/apps/rmc-ab-initio/configs (e.g. ladder-lif-node).
export def run [
    grid: int = 20
    --config: string = "ladder-lif-node"
    --rank: int = 0               # slurm-style rank (chain offset = rank · chains)
    --rank-table: int = 800       # table rank in the artifact filename
    --lanes: int = 32
    --max-order-gpu: int = 256
    --out-dir: string = "results-gpu"
] {
    cuda load
    cd $ROOT
    let table = (table-path $grid $rank_table)
    let config_path = if ($config | path exists) {
        $config
    } else {
        $ROOT | path join crates apps rmc-ab-initio configs $"($config).json"
    }
    (cargo run --release -p rmc-ab-initio --features gpu-cuda --
        run-gpu --table $table --config $config_path --rank $rank
        --lanes $lanes --max-order-gpu $max_order_gpu --out-dir $out_dir)
}

# A5 CUDA parity checkpoint: the engine parity suite on this node's GPU.
export def parity [] {
    cuda load
    cd $ROOT
    cargo test --release -p rmc-ab-initio --features gpu-cuda --test gpu_engine_parity -- --nocapture
}

# nvidia-smi snapshot of the visible devices (sanity check inside jobs).
export def info [] {
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
}
