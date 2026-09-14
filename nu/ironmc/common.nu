# Shared paths and helpers for the ironmc developer module.

export const ROOT = (path self | path dirname --num-levels 3)
export const DATASET = ($ROOT | path join datasets LiF)
export const CONFIGS = ($ROOT | path join crates apps rmc-ab-initio configs)

# Canonical table artifact path (matches the D7 naming convention).
export def table-path [grid: int, rank: int = 800, precision: string = "f32"] {
    $DATASET | path join $"table-n($grid)-r($rank)-($precision).rmcvtab"
}

# Config names from the rmc-ab-initio configs dir, for `--config: string@config-names`.
export def config-names [] {
    ls $CONFIGS | get name | path basename
        | where {|file| $file | str ends-with ".json" }
        | each {|file| $file | str replace ".json" "" }
}
