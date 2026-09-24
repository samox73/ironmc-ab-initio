# Shared paths and helpers for the ironmc developer module.

export const ROOT = (path self | path dirname --num-levels 3)
export const DATASET = ($ROOT | path join datasets LiF)
export const CONFIGS = ($ROOT | path join crates apps rmc-ab-initio configs)

# Canonical table artifact path (matches the D7 naming convention).
export def table-path [grid: int, rank: int = 800, precision: string = "f32"] {
    $DATASET | path join $"table-n($grid)-r($rank)-($precision).rmcvtab"
}

# One summary row per results/merged.json found at or below each run dir/glob
# (the engine behind `ironmc results` and `ironmc ladder fit <runs>`).
# grid comes from the sibling run-metadata.json (written by the binary; for
# runs predating it, regenerate with the metadata-only config — see
# configs/README.md); null when the file is missing.
export def run-results [
    ...runs: string   # run dirs or globs (*, **, {a,b}), relative to repo root; default: all under runs/
] {
    let patterns = if ($runs | is-empty) { ["runs"] } else { $runs }
    let files = $patterns | each {|p|
        # wax's ** needs ≥1 path segment and rejects an adjacent **/**, so strip a
        # trailing ** and re-append the deep-find tail ourselves.
        let base = ($p | str trim --right --char '/' | str replace --regex '\*\*$' '' | str trim --right --char '/')
        let root = (if ($base | str starts-with '/') { $base } else { $ROOT | path join $base })
        glob ($root | path join "**" merged.json)
    } | flatten | uniq | sort
    $files | each {|f|
        let r = (open $f)
        let meta = ($f | path dirname | path join run-metadata.json)
        {
            run: ($f | path dirname | path dirname | str replace $"($ROOT | path join runs)/" "")
            grid: (if ($meta | path exists) { open $meta | get -o grid } else { null })
            e0: $r.e0_thermo.mean
            e0_err: $r.e0_thermo.stderr
            sign: $r.sign.mean
            sign_err: $r.sign.stderr
            order: $r.mean_order.mean
            order_err: $r.mean_order.stderr
            z: ($r | get -o z.mean)
            z_err: ($r | get -o z.stderr)
            steps: $r.steps_done
        }
    } | sort-by run
}

# Existing run dirs (relative to the repo root) for `--out-dir: string@run-dirs`
# completion — resume, merge, and annotation targets. New run names are typed
# freely; jobs/ internals are noise, not targets.
export def run-dirs [] {
    glob --no-file ($ROOT | path join runs **) --exclude [**/jobs **/jobs/**]
        | each {|d| $d | path relative-to $ROOT }
        | sort
}

# Config names from the rmc-ab-initio configs dir, for `--config: string@config-names`.
export def config-names [] {
    ls $CONFIGS | get name | path basename
        | where {|file| $file | str ends-with ".json" }
        | each {|file| $file | str replace ".json" "" }
}
