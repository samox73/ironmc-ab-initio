# Polaron dispersion E₀(p) sweeps along the Γ→X and Γ→L legs (see
# latex/symmetry-paths/symmetry-paths.tex for why these two directions).
#
#   ironmc dispersion sweep 20 --tau 60 --l-js [1 2 3]   # scout: full Γ→X + 3 Γ→L points
#   ironmc dispersion sweep 40 --x-js [1 2 3 4 5 6]      # ladder window, τ240 default
#   ironmc dispersion points 20                          # inspect the point list, no submission
#   ironmc results runs/dispersion/n20-tau60             # tabulate finished points
#
# One slurm array task per k-point: each task gets one a100 node (2 GPUs = 2
# ranks on the same point) and its own run dir
# runs/dispersion/n<grid>-tau<τ>/results/<label>, so merge never sees mixed
# ε(p_ext) and checkpoint/resume works per point. The label encodes the point
# (x<j> = (j/n, 0, j/n), l<j> = (j/n, j/n, j/n)); the experiment dir encodes
# grid and τ — that keeps the in-job closure argless and self-contained.
# Γ (j = 0) is deliberately absent: every grid/τ already has its Γ point in
# runs/e0-fit; the fit runbook reuses those.
#
# Resubmit the same sweep until every point's merged.json says complete —
# finished points no-op in seconds and free their node for pending tasks.

use common.nu [ROOT CONFIGS table-path]
use slurm.nu
use gpu.nu
use cpu.nu

# label ("x3" / "l2") → fractional p on the n³ grid. Γ→X is a Cartesian
# ⟨100⟩ axis (|p| = 2j/n · 2π/a), Γ→L the ⟨111⟩ diagonal (|p| = √3·j/n · 2π/a).
def decode [label: string, n: int] {
    let m = ($label | parse --regex '^(?<leg>[xl])(?<j>\d+)$')
    if ($m | is-empty) {
        error make {msg: $"bad point label ($label): expected x<j> \(Γ→X\) or l<j> \(Γ→L\)"}
    }
    let j = ($m.0.j | into int)
    if $j < 1 or $j > ($n // 2) {
        error make {msg: $"point ($label) is off the leg on the ($n)³ grid: need 1 ≤ j ≤ (($n // 2)) — j = 0 is the Γ point, reuse runs/e0-fit"}
    }
    let f = ($j / $n)
    if $m.0.leg == "x" { [$f, 0.0, $f] } else { [$f, $f, $f] }
}

# The point list a sweep would run: label, leg, fractional p, and |p| in
# units of 2π/a — pipe this into the fit runbook or eyeball it before a sweep.
export def points [
    grid: int
    --x-js: list<int>       # Γ→X indices (default 1..min(10, n/2))
    --l-js: list<int> = []  # Γ→L indices (default none — scout passes [1 2 3])
] {
    let x_js = ($x_js | default (1..([10 ($grid // 2)] | math min) | each {|j| $j }))
    let pts = (
        ($x_js | each {|j| {label: $"x($j)", leg: "Γ→X", j: $j} })
        | append ($l_js | each {|j| {label: $"l($j)", leg: "Γ→L", j: $j} })
    )
    $pts | each {|pt|
        let p = (decode $pt.label $grid)
        $pt | insert p $p | insert p_2pi_a (
            if $pt.leg == "Γ→X" { 2 * $pt.j / $grid } else { 1.7320508075688772 * $pt.j / $grid }
        )
    }
}

def sweep-config [tau: int, cpu: bool] {
    let base = if $cpu { "lif-slotmap" } else { "lif-gpu" }
    if $tau == 240 { $base } else { $"($base)-tau($tau)" }
}

# Submit the sweep as one slurm array job on the devel QoS: one task per
# k-point — 1 a100 node (2 GPUs = 2 ranks) by default, or with --cpu 5 zen3
# nodes (5 slotmap ranks) per point for grids whose vtab plane exceeds the
# GPU's per-allocation cap (n = 120 needs 33 GB vs the A100's 10.6 GB — same
# reason the e0-fit n100/n120 rungs ran on CPU). Pick the engine once per
# experiment: the two checkpoint formats are not interchangeable mid-run.
# Resubmit the identical command to resume.
export def sweep [
    grid: int
    --tau: int = 240        # production default; 60 for the scout (config lif-{gpu,slotmap}-tau<τ>)
    --x-js: list<int>       # Γ→X indices (default 1..min(10, n/2))
    --l-js: list<int> = []  # Γ→L indices (default none)
    --cpu                   # slotmap engine on zen3 nodes instead of the GPU
    --ranks: int            # MC ranks per point (default: 2 = both GPUs, or 5 CPU nodes)
    --rounds: int = 1       # queue this many resubmission rounds up front, each chained
                            # --dependency=afterany behind the previous (a GPU τ240 point
                            # needs ~6-8 devel legs; completed points no-op). Overshooting
                            # costs only short no-op tasks.
    --dry-run               # print the batch script instead of submitting (single round)
] {
    let pts = if $x_js == null { points $grid --l-js $l_js } else { points $grid --x-js $x_js --l-js $l_js }
    let labels = ($pts | get label)
    let config = (sweep-config $tau $cpu)
    if not ($CONFIGS | path join $"($config).json" | path exists) {
        error make {msg: $"no config ($config).json for τ = ($tau) — only τ with a ($config | str replace --regex '-tau.*' '')[-tauN] config can sweep"}
    }
    let out_dir = $"runs/dispersion/n($grid)-tau($tau)"
    let partition = if $cpu { "zen3_0512" } else { "zen3_0512_a100x2" }
    let ranks = ($ranks | default (if $cpu { 5 } else { 2 }))
    # The job closure is stored, not inlined: `view source` still serializes its
    # literal text, and one variable beats four submit-call branches.
    let job = if $cpu { { ironmc dispersion point --cpu } } else { { ironmc dispersion point } }
    if $dry_run {
        print ($pts | to nuon)
        return (slurm submit $job --array-dirs $labels --out-dir $out_dir
            --partition $partition --devel --ranks $ranks --dry-run)
    }
    if not (table-path $grid 800 | path exists) {
        error make {msg: $"no table at (table-path $grid 800) — build/promote it first \(ironmc tables\)"}
    }
    if $rounds < 1 { error make {msg: "--rounds must be ≥ 1"} }
    mut dep = ""
    mut submissions = []
    for round in 1..$rounds {
        let r = if $dep == "" {
            # --chain: a fresh sweep submits normally; re-running while earlier
            # rounds are still queued appends behind their tail instead of erroring.
            (slurm submit $job --array-dirs $labels --out-dir $out_dir
                --partition $partition --devel --ranks $ranks --chain)
        } else {
            (slurm submit $job --array-dirs $labels --out-dir $out_dir
                --partition $partition --devel --ranks $ranks --after $dep)
        }
        if $r.job_id == null {
            error make {msg: $"round ($round): sbatch returned no job id — cannot chain the remaining rounds"}
        }
        $dep = $r.job_id
        $submissions = ($submissions | append ($r | insert round $round))
        # jobdir names are stamped to the second; don't let two rounds collide.
        if $round < $rounds { sleep 1sec }
    }
    $submissions
}

# In-job target for one array task: everything is decoded from
# $IRONMC_OUT_DIR = …/runs/dispersion/n<grid>-tau<τ>/results/<label>, which
# `slurm submit --array-dirs` makes point-specific per task.
export def point [
    --cpu   # slotmap engine (must match the sweep that created the experiment)
] {
    let out_dir = $env.IRONMC_OUT_DIR?
    if $out_dir == null {
        error make {msg: "dispersion point runs inside a `dispersion sweep` array job ($IRONMC_OUT_DIR is unset)"}
    }
    let m = ($out_dir | parse --regex 'n(?<n>\d+)-tau(?<tau>\d+)[/\\]results[/\\](?<label>[a-z0-9]+)$')
    if ($m | is-empty) {
        error make {msg: $"cannot decode grid/τ/point from ($out_dir) — expected …/runs/dispersion/n<grid>-tau<τ>/results/<label>"}
    }
    let grid = ($m.0.n | into int)
    let tau = ($m.0.tau | into int)
    let p = (decode $m.0.label $grid)
    let config = (sweep-config $tau $cpu)
    print $"dispersion point ($m.0.label) on ($grid)³ τ($tau) \(($config)\): p_ext = ($p)"
    if $cpu {
        cpu run $grid --config $config --p-ext $p
    } else {
        gpu run $grid --config $config --p-ext $p
    }
}
