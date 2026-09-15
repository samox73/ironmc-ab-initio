# CPU (slotmap engine, rayon) targets — the fast CPU production path and the
# CPU twin of `gpu run`.
#
#   ironmc cpu run 20 --out-dir results/cpu/n20-prod
#   ironmc slurm submit { ironmc cpu run 50 --out-dir results/cpu/n50-prod } --partition zen3_0512 --devel --ranks 5
#
# Same RankDump/Checkpoint/merge machinery as `gpu run`: resubmitting with the
# same --out-dir resumes the run bit-identically, and independent --rank ranks
# across nodes merge offline. This uses the slotmap engine (O(1) diagram updates,
# ~20× the per-core throughput of the batched `FlatDiagram` engine). The batched
# engine remains available as the `run-flat` subcommand — the GPU-parity
# reference — but is not exposed here.

use common.nu [ROOT config-names]

# CPU run of a DiagConfig on this node's cores; writes a RankDump + Checkpoint
# into --out-dir for resume and `merge`. `config` may be a path or a name from
# crates/apps/rmc-ab-initio/configs (e.g. ladder-lif-slotmap). ladder-lif-slotmap
# and ladder-lif-flat are kept identical for a clean 1:1 engine comparison.
export def run [
    grid: int = 20                # table grid n → datasets/LiF/table-n{n}-r{rank}-f32.rmcvtab
    --config: string@config-names = "ladder-lif-slotmap"
    --rank: int                   # chain-offset rank; defaults to $SLURM_PROCID (multi-rank srun)
    --chains: int = 0             # override the config's chain count (0 = keep config; must be pinned for multi-rank)
    --rank-table: int = 800       # table rank in the artifact filename
    --out-dir: string             # REQUIRED run directory (e.g. results/cpu/n50-prod): holds
                                  # dumps + checkpoints; resubmitting with the same dir resumes
                                  # the run (bit-identical to an uninterrupted one).
] {
    if $out_dir == null {
        error make {msg: "every run must be named: pass --out-dir results/cpu/<run-name>"}
    }
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
    (cargo run --release -p rmc-ab-initio --
        run-cpu --table $table --config $config_path --rank $rank --chains $chains
        --out-dir $out_dir)
}

# Batched FlatDiagram CPU run (`run-flat` subcommand) — the GPU-parity reference
# engine, exposed here only to benchmark it head-to-head against `cpu run`
# (slotmap) at the same --ranks. Same RankDump/merge, but a batched checkpoint
# (not interchangeable with `cpu run`'s — use a separate --out-dir).
export def flat [
    grid: int = 20                # table grid n → datasets/LiF/table-n{n}-r{rank}-f32.rmcvtab
    --config: string@config-names = "ladder-lif-flat"
    --rank: int                   # chain-offset rank; defaults to $SLURM_PROCID
    --chains: int = 0             # override the config's chain count (0 = keep config)
    --rank-table: int = 800       # table rank in the artifact filename
    --group: int = 1              # update-selection group size (1 = per-chain)
    --max-order-gpu: int = 256    # FlatDiagram capacity = 2·N+2
    --leg-cycles: int = 0         # cap this leg at N cycles (0 = run to completion/deadline)
    --out-dir: string             # REQUIRED run directory (holds dumps + a batched checkpoint)
] {
    if $out_dir == null {
        error make {msg: "every run must be named: pass --out-dir results/cpu/<run-name>"}
    }
    cd $ROOT
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
    (cargo run --release -p rmc-ab-initio --
        run-flat --table $table --config $config_path --rank $rank --chains $chains
        --group $group --max-order-gpu $max_order_gpu
        --leg-cycles $leg_cycles --out-dir $out_dir)
}

# Merge every RankDump (all ranks, all legs) in a run dir into one report —
# engine-agnostic (CPU and GPU dumps share the format). The cumulative
# statistics of a checkpointed/multi-rank run.
export def merge [
    dir: string                   # e.g. results/cpu/n50-prod
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
