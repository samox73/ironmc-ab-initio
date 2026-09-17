# IronMC developer commands — a nushell module instead of a Makefile.
#
# Load (from the repo root):        use nu/ironmc
# or globally in your config.nu:    use /home/samox/studies/phd/repositories/ironmc/nu/ironmc
# Flat names instead of prefixed:   use nu/ironmc *
#
# Everyday flow for the LiF results:
#   ironmc status                       # what exists, where things stand
#   ironmc test lif                     # dataset validation ladder
#   ironmc gates e2; ironmc gates dmc   # compression gates
#   ironmc tables ladder                # build missing production tables
#   ironmc ladder checkpoint            # 20³ vs FEP-DMC (stage 2)
#   ironmc ladder guard-tau --baseline <e0>
#   ironmc ladder guard-rank --baseline <e0>
#   ironmc ladder sweep | ironmc ladder fit
#   ironmc results                      # e0/sign/order table of finished runs

use common.nu [ROOT DATASET run-results]
export use tables.nu
export use ladder.nu
export use gates.nu
export use test.nu
export use bench.nu
export use qe.nu
export use slurm.nu
export use cuda.nu
export use configs.nu
export use cpu.nu
export use gpu.nu
export use frohlich.nu

# Result summaries of finished runs: one row per merged.json — the grid n
# (from the run's run-metadata.json; null for runs predating it), e0_thermo,
# sign, and mean order with stderrs, plus the decay-fit e0/z when the run
# produced them (free-τ mode only; fixed-τ runs omit those fields).
#
# Each argument is a dir or glob, resolved relative to the repo root, and we
# deep-find every results/merged.json at or below it — so all of these work:
#   ironmc results                              # every run under runs/
#   ironmc results runs/guards/n20-tau240       # one run dir
#   ironmc results runs/guards/*                # every run under guards/
#   ironmc results runs/**                      # every run, nested included
#   ironmc results runs/guards/{n20-r1600,n20-tau240}   # brace alternation
export def results [
    ...runs: string   # run dirs or globs (*, **, {a,b}), relative to repo root; default: all under runs/
] {
    run-results ...$runs
}

# One-screen overview: built tables and the latest journal entries.
export def status [] {
    {
        tables: (ls $DATASET | where name =~ '\.rmcvtab$' | select name size)
        journal: (
            open --raw ($ROOT | path join journal.md) | lines
                | where {|line| $line | str starts-with "## " } | last 4
        )
    }
}
