# Vertex-table generation (Plan D5 CLI wrappers).
#
# Sharded builds (large n within a 10-min devel leg). The n³ grid splits into S
# balanced slabs, one per rank; each rank fills its slab, shards resume across
# legs, and `merge-shards` reassembles the final .rmcvtab (applying the global
# band-edge shift once). Under slurm the shards and the merged table are written
# to the run's results dir ($IRONMC_OUT_DIR = <out-dir>/results), NOT datasets/ —
# so a build can't clobber a canonical table. Once validated, `ironmc tables
# promote <n>` copies the finished .rmcvtab into datasets/LiF.
#
# `slurm submit` auto-derives --out-dir = runs/tables/n<grid> for these table
# targets, so a grid's shard and merge jobs share a dir with nothing to repeat.
#
# CPU-fill recipe — zen3_0512_devel has 5 nodes, so use all 5; --shards
# auto-matches --ranks, and merge discovers the count on disk:
#
#   ironmc slurm submit { ironmc tables shard 120 } --partition zen3_0512 --devel --ranks 5
#   # resubmit the same line until every shard reports "already built" (resume), then:
#   ironmc slurm submit { ironmc tables merge-shards 120 } --devel
#   # → runs/tables/n120/results/table-n120-r800-f32.rmcvtab
#
# GPU fill — one shard per A100 (4 GPUs across the 2 devel nodes). `--device gpu`
# runs the gpu-cuda binary (the Fourier fill on the A100s; base data stays CPU),
# which is exactly what a GPU-partition submit prebuilds — no in-job rebuild:
#
#   ironmc slurm submit { ironmc tables shard 120 --device gpu } --partition zen3_0512_a100x2 --devel --ranks 4
#   ironmc slurm submit { ironmc tables merge-shards 120 } --devel
#
# The derived dir is device-independent (runs/tables/n120 for both), so a grid's
# GPU and CPU builds share it — pass an explicit --out-dir to keep them apart.
# Lower $env.RMC_GPU_CHUNK (default 16384 points/launch) if a shard OOMs the GPU.
# CPU fill (--device cpu, the default) still runs best on zen3_0512's 5 nodes: a
# GPU-partition submit prebuilds only gpu-cuda, so a CPU `cargo run` there would
# recompile in-job.
#
# Validate a merge without clobbering an existing table by directing it elsewhere:
#   ironmc tables build 65 --out /tmp/ref.rmcvtab   # single-node reference
#   ironmc tables merge-shards 65 --out /tmp/merged.rmcvtab
#   (open --raw /tmp/ref.rmcvtab | hash sha256) == (open --raw /tmp/merged.rmcvtab | hash sha256)

use common.nu [ROOT CUDA_TARGET_DIR DATASET table-path]
use cuda.nu

# Where sharded builds read/write their table files: the slurm run's results dir
# ($IRONMC_OUT_DIR, set by `slurm submit --out-dir`) when in a job, else
# datasets/LiF for a local build. Keeps job output under runs/.../results so a
# build can't accidentally overwrite a canonical table in datasets/ — promote
# the finished .rmcvtab into datasets/LiF (`tables promote`) once validated.
def out-base [] {
    $env.IRONMC_OUT_DIR? | default $DATASET
}

# Build one vertex table. Defaults are the certified production settings.
export def build [
    grid: int              # q/k grid size n (table covers n³ points)
    --rank: int = 800      # global SVD budget (1600 = untruncated reference)
    --precision: string = "f32"
    --max-gib: int         # override the memory guard (deliberate decision!)
    --out: string          # output path (default: <results-or-datasets>/table-...rmcvtab,
                           # like shard/merge-shards); pass a temp path for a
                           # no-clobber reference build
] {
    # Same out-base as shard/merge-shards: under slurm the table lands in the
    # run's results dir, never straight onto the canonical one in datasets/
    # (`tables promote` moves it there once validated).
    let base = (out-base)
    mkdir $base
    let out = ($out | default ($base | path join $"table-n($grid)-r($rank)-($precision).rmcvtab"))
    cd $ROOT
    if $max_gib == null {
        cargo run --release -p rmc-ab-initio -- table --grid $grid --total-rank $rank --precision $precision --out $out
    } else {
        cargo run --release -p rmc-ab-initio -- table --grid $grid --total-rank $rank --precision $precision --out $out --max-gib $max_gib
    }
    $out
}

# Build one shard of a table's grid. --shards and --rank default to the srun
# environment ($SLURM_NTASKS / $SLURM_PROCID), so `--ranks S` fills all S shards
# in one job with no explicit counts. An already-built shard is a no-op — resubmit
# the same line to resume across devel legs.
export def shard [
    grid: int                     # q/k grid size n (table covers n³ points)
    --shards: int                 # number of slabs; defaults to $SLURM_NTASKS (auto-matches --ranks)
    --rank: int                   # shard index; defaults to $SLURM_PROCID (multi-rank srun)
    --rank-table: int = 800       # global SVD budget (in the artifact filename)
    --precision: string = "f32"
    --device: string = "cpu"      # "gpu" moves the Fourier fill to the A100s (needs a GPU partition)
    --max-gib: int                # override the per-shard memory guard
] {
    let shards = ($shards | default ($env.SLURM_NTASKS? | default 4 | into int))
    let rank = ($rank | default ($env.SLURM_PROCID? | default 0 | into int))
    let base = (out-base)
    mkdir $base
    let out = ($base | path join $"table-n($grid)-r($rank_table)-($precision).rmcvtab")
    cd $ROOT
    let max_args = if $max_gib == null { [] } else { [--max-gib $max_gib] }
    # A GPU fill needs the gpu-cuda binary — build/run that exact feature set so it
    # matches what `slurm submit` prebuilds on a GPU partition (no in-job rebuild).
    # `cuda load` first: libnvrtc is dlopened at run time and cudarc reads
    # $CUDA_PATH, neither of which the batch script provides on its own.
    let feat_args = if $device == "gpu" { cuda load; [--features gpu-cuda --target-dir $CUDA_TARGET_DIR] } else { [] }
    (cargo run --release -p rmc-ab-initio ...$feat_args --
        table --grid $grid --total-rank $rank_table --precision $precision --device $device
        --shard $rank --shards $shards --out $out ...$max_args)
    $"($out).shard-($rank)-of-($shards)"
}

# Reassemble a full table from its shards. The shard count is discovered from the
# files on disk (no --shards to keep in sync with the fill job — the merge runs as
# a single task, so $SLURM_NTASKS would be 1 there). The Rust side still refuses a
# set that isn't the complete contiguous partition.
export def merge-shards [
    grid: int
    --rank-table: int = 800
    --precision: string = "f32"
    --out: string                 # output path (default: <results-or-datasets>/table-...rmcvtab);
                                  # pass a temp path to merge without clobbering an existing table
] {
    # Read the shards from — and write the merged table into — the same dir the
    # shard job used (the run's results/ under slurm, else datasets/LiF), so a
    # merge job with the SAME --out-dir as its shard job finds them.
    let base = (out-base)
    let table = ($base | path join $"table-n($grid)-r($rank_table)-($precision).rmcvtab")
    cd $ROOT
    let shard_files = (glob $"($table).shard-*-of-*")
    if ($shard_files | is-empty) {
        error make {msg: $"no shard files matching ($table).shard-*-of-*"}
    }
    # One shard set only: mixed -of-N (e.g. a stale of-4 beside a fresh of-5) would
    # be a confusing partial-set error downstream — catch it here with a clear msg.
    let sets = ($shard_files | each {|f| $f | parse --regex 'of-(?<n>\d+)$' | get -o 0.n } | uniq)
    if ($sets | length) > 1 {
        error make {msg: $"multiple shard sets present \(of-($sets | str join ', ')\); remove the stale one"}
    }
    let out = ($out | default $table)
    cargo run --release -p rmc-ab-initio -- table-merge --out $out ...$shard_files

    $out
}

# Build every missing production ladder table (20³..50³ at rank 800/f32).
export def ladder [--rank: int = 800] {
    [20 32 40 50] | each {|n|
        # Same path build resolves to, so the existed-check matches what a
        # rerun would actually write (results dir under slurm, else datasets).
        let out = ((out-base) | path join $"table-n($n)-r($rank)-f32.rmcvtab")
        let existed = ($out | path exists)
        if not $existed { build $n --rank $rank }
        {grid: $n, table: $out, built: (not $existed)}
    }
}

# The untruncated reference table for rank guards (rank 1600).
export def reference [grid: int = 20] {
    build $grid --rank 1600
}

# Promote a validated table from its run's results dir into datasets/LiF, where
# `cpu run`/`gpu run` look for it (they hardcode the canonical table-path). The
# source defaults to the dir `slurm submit` derives for table jobs.
export def promote [
    grid: int
    --rank-table: int = 800
    --precision: string = "f32"
    --from: string                # source dir (default runs/tables/n<grid>/results)
    --force                       # overwrite an existing canonical table
] {
    let data_root = ($env.IRONMC_DATA_ROOT? | default ($ROOT | path expand))
    let from = ($from | default ($data_root | path join runs tables $"n($grid)" results))
    let name = $"table-n($grid)-r($rank_table)-($precision).rmcvtab"
    let src = ($from | path join $name)
    if not ($src | path exists) {
        error make {msg: $"no merged table at ($src) — run `tables merge-shards ($grid)` first"}
    }
    let dest = ($DATASET | path join $name)
    if ($dest | path exists) and not $force {
        error make {msg: $"($dest) already exists — pass --force to replace the canonical table"}
    }
    cp $src $dest
    print $"promoted ($src) → ($dest)"
    $dest
}

# List built table artifacts with sizes.
export def list [] {
    ls $DATASET | where name =~ '\.rmcvtab$' | select name size modified
}
