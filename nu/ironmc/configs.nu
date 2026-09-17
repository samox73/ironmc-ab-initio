# Run-config (DiagConfig JSON) helpers.
#
# Naming: lif-<engine>[-tauN].json — no τ suffix means the τ=240 production
# default (≈ FEP-DMC's effective 1/k_BT at 50 K); τ60/τ120 variants carry
# their max_tau in the name.
#
#   ironmc configs list               # one row per config, key knobs side by side
#   ironmc configs list lif-gpu lif-gpu-tau60
#   ironmc configs edit lif-slotmap   # open in $EDITOR

use common.nu [CONFIGS config-names]

# Key sampling knobs of every config (or the named ones), one row each.
export def list [
    ...names: string@config-names   # default: all configs
] {
    let names = if ($names | is-empty) { config-names } else { $names }
    $names | each {|n|
        let path = ($CONFIGS | path join $"($n).json")
        open $path | insert config $n | insert path $path
    } | select -o config max_tau tau_fit chains max_steps warmup_steps max_order n_batches seed fixed_tau path
}

# Open a config in $EDITOR.
export def edit [
    name: string@config-names
] {
    let editor = ($env.EDITOR? | default "vi" | split row " ")
    run-external ($editor | first) ...($editor | skip 1) ($CONFIGS | path join $"($name).json")
}

# Print one config in full.
export def show [
    name: string@config-names
] {
    open ($CONFIGS | path join $"($name).json")
}
