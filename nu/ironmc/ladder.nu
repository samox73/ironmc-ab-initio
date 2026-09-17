# D7 grid-ladder runs, guards, and the 1/n → 0 extrapolation.
#
# Every command returns structured records — pipe them, don't parse them:
#   ironmc ladder run 20
#   ironmc ladder fit runs/e0-fit/tau240

use common.nu [ROOT CONFIGS table-path run-results]
use tables.nu

# FEP-DMC's LiF(e) 20³ reference binding (Etrue − Ep, their example output).
export const FEPDMC_20 = -0.2479

def run-table [table: string, config: string] {
    cd $ROOT
    cargo run --release -p rmc-ab-initio -- run --table $table --config $config | from json
}

# Private core: internal callers use this name so that environments where a
# global `run` alias/keyword shadows bare calls (e.g. cluster configs) cannot
# break module parsing.
def point [
    grid: int
    --rank: int = 800
    --tau120
    --node          # full-node config: chains auto-sized to the CPU allocation, 10⁷ steps each
    --max-steps: int = 0  # override the config's per-chain max_steps (0 = use the config),
                          # e.g. oversized when relying on the SLURM end-time deadline.
                          # Any oversize factor works with multi-chain configs: the
                          # jackknife pools batches across chains (≥1 per chain), so a
                          # truncated run still has ≥ chains blocks.
] {
    let config = if $tau120 and $node {
        $CONFIGS | path join lif-node-tau120.json
    } else if $tau120 {
        $CONFIGS | path join lif-tau120.json
    } else if $node {
        $CONFIGS | path join lif-node-tau60.json
    } else {
        $CONFIGS | path join lif-tau60.json
    }
    let config = if $max_steps > 0 {
        let patched = (mktemp -t --suffix .json)
        open $config | update max_steps $max_steps | save -f $patched
        $patched
    } else { $config }
    let report = (run-table (table-path $grid $rank) $config)
    {
        grid: $grid
        rank: $rank
        tau: (if $tau120 { 120 } else { 60 })
        e0: $report.e0_thermo.mean
        stderr: $report.e0_thermo.stderr
        sign: $report.sign.mean
        order: $report.mean_order.mean
        steps_done: $report.steps_done
    }
}

# Run one ladder point; returns {grid, rank, e0, stderr, sign, order, steps_done}.
export def run [
    grid: int
    --rank: int = 800
    --tau120           # τ-doubling variant (guard A)
    --node             # full-node config (chains = allocated CPUs) for slurm jobs
    --max-steps: int = 0  # override the config's per-chain max_steps (0 = use the config)
] {
    if $tau120 and $node {
        point $grid --rank $rank --max-steps $max_steps --tau120 --node
    } else if $tau120 {
        point $grid --rank $rank --max-steps $max_steps --tau120
    } else if $node {
        point $grid --rank $rank --max-steps $max_steps --node
    } else {
        point $grid --rank $rank --max-steps $max_steps
    }
}

# Stage-2 checkpoint: E₀(20³) vs FEP-DMC, ±30 meV band.
export def checkpoint [] {
    let r = (point 20)
    let diff_mev = (($r.e0 - $FEPDMC_20) * 1000)
    $r | insert fepdmc $FEPDMC_20 | insert diff_mev $diff_mev
       | insert pass (($diff_mev | math abs) < 30)
}

# Guard A: τ-doubling at 20³. Pass --baseline <e0> to reuse an existing τ=60
# result instead of rerunning it.
export def guard-tau [--baseline: float, --baseline-stderr: float = 0.0003] {
    let base = if $baseline == null { point 20 } else {
        {e0: $baseline, stderr: $baseline_stderr}
    }
    let doubled = (point 20 --tau120)
    let shift = (($base.e0 - $doubled.e0) | math abs)
    let tolerance = (2 * ($base.stderr ** 2 + $doubled.stderr ** 2 | math sqrt))
    {
        tau60: $base.e0
        tau120: $doubled.e0
        shift_mev: ($shift * 1000)
        tolerance_mev: ($tolerance * 1000)
        pass: ($shift < $tolerance)
    }
}

# Guard B: Δ_rank(800) on the production grid — builds the untruncated
# reference table if missing, runs it, and compares. The resulting delta is the
# quoted compression systematic for the whole ladder (gate: max(2σ, 3 meV)).
export def guard-rank [
    grid: int = 20
    --baseline: float            # e0 from an existing rank-800 run at this grid
    --baseline-stderr: float = 0.0003
] {
    if not (table-path $grid 1600 | path exists) { tables reference $grid }
    let base = if $baseline == null { point $grid } else {
        {e0: $baseline, stderr: $baseline_stderr}
    }
    let reference = (point $grid --rank 1600)
    let delta = (($base.e0 - $reference.e0) | math abs)
    let tolerance = ([(2 * ($base.stderr ** 2 + $reference.stderr ** 2 | math sqrt)) 0.003] | math max)
    {
        e0_800: $base.e0
        e0_1600: $reference.e0
        delta_rank_mev: ($delta * 1000)
        tolerance_mev: ($tolerance * 1000)
        pass: ($delta < $tolerance)
    }
}

# The remaining production points (run after both guards pass).
export def sweep [--node] {
    [32 40 50] | each {|n| if $node { point $n --node } else { point $n } }
}

# Linear fit of E₀ against 1/n over finished ladder runs.
# The intercept is the thermodynamic-limit estimate (FEP-DMC: ≈ −0.408 eV).
#
# Point the fit at existing results — the grid n comes from each run's
# run-metadata.json (written by the binary; annotate older runs with the
# metadata-only config, see configs/README.md):
#   ironmc ladder fit runs/e0-fit/tau240
#   ironmc ladder fit runs/e0-fit/tau240/{n40,n60,n80}   # a subset of points
#
# Piped input works too: `ironmc results ...` rows, or bare {grid, e0} records:
#   ironmc results runs/e0-fit/tau240 | where e0_err < 0.001 | ironmc ladder fit
#   ironmc ladder sweep | ironmc ladder fit
export def fit [
    ...runs: string   # run dirs or globs (*, **, {a,b}), relative to repo root; when given, piped input is ignored
] {
    let rows = if ($runs | is-empty) { $in } else { run-results ...$runs }
    let pts = (
        $rows | each {|r|
            if ($r | get -o grid) == null {
                error make {msg: $"no grid for ($r | get -o run | default ($r | to nuon)): the run has no run-metadata.json — annotate it with `ironmc cpu run <n> --config metadata-only --out-dir <run>/results`"}
            }
            { grid: $r.grid, e0: $r.e0 }
        } | sort-by grid | insert x {|r| 1.0 / $r.grid }
    )
    let xm = ($pts.x | math avg)
    let ym = ($pts.e0 | math avg)
    let slope = (
        ($pts | each {|p| ($p.x - $xm) * ($p.e0 - $ym) } | math sum)
        / ($pts | each {|p| ($p.x - $xm) ** 2 } | math sum)
    )
    {
        slope: $slope
        intercept: ($ym - $slope * $xm)
        points: ($pts | select grid x e0)
    }
}
