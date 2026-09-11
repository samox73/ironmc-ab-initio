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

use common.nu [ROOT DATASET]
export use tables.nu
export use ladder.nu
export use gates.nu
export use test.nu
export use bench.nu
export use qe.nu
export use slurm.nu
export use cuda.nu
export use gpu.nu
export use frohlich.nu

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
