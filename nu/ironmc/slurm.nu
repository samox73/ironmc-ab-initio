# Dispatch ironmc targets as slurm jobs on VSC-5.
#
#   ironmc slurm submit ladder run 50 --time 08:00:00
#   ironmc slurm submit gates dmc --devel              # 10-min devel QoS
#   ironmc slurm submit tables build 50 --dry-run      # inspect, don't submit
#   (quote the target only if it carries its own flags:
#    ironmc slurm submit "ladder guard-tau --baseline -0.25")
#
# submit prebuilds the whole workspace in release mode before calling sbatch
# (jobs share $DATA, so compiled artifacts carry over and the job never spends
# walltime compiling). --dry-run skips the build.

use common.nu [ROOT]

# VSC-5 queue map (gitlab.tuwien.ac.at vsc5_queues.md, fetched 2026-09-07).
export const PARTITIONS = [
    [partition, default_qos, devel_qos, max_walltime, description];
    [zen3_0512, zen3_0512, zen3_0512_devel, "72:00:00", "default, 512 GB"]
    [zen3_1024, zen3_1024, null, "72:00:00", "high memory, 1 TB"]
    [zen3_2048, zen3_2048, null, "72:00:00", "higher memory, 2 TB"]
    [zen2_0256_a40x2, zen2_0256_a40x2, null, "72:00:00", "GPU, 2× A40"]
    [zen3_0512_a100x2, zen3_0512_a100x2, zen3_0512_a100x2_devel, "72:00:00", "GPU, 2× A100"]
]

const DEVEL_WALLTIME = "00:10:00"

# Dispatchable targets = the module's own commands, discovered from scope, so
# the list stays current automatically. The completer is context-aware and
# completes ONE word at a time (some nu builds insert multi-word completion
# values unquoted), so `submit gates <Tab>` offers e2/dmc, etc.
def target-completions [context: string] {
    let all = (scope commands
        | where name =~ '^(ironmc )?(tables|ladder|gates|test|bench|qe) '
        | each {|cmd|
            {
                name: ($cmd.name | str replace "ironmc " "")
                description: $cmd.description
            }
        }
        | uniq-by name)

    # Target words already committed after `submit` (the in-progress token is
    # dropped — nu filters candidates by it — and flags end the target).
    let tail = ($context | str replace --regex '.*\bsubmit\s*' "")
    mut words = ($tail | split row " " | where {|word| $word != "" })
    if not ($context | str ends-with " ") {
        $words = ($words | drop 1)
    }
    let words = ($words | take while {|word| not ($word | str starts-with "-") })
    let prefix = if ($words | is-empty) { "" } else { ($words | str join " ") + " " }
    let position = ($words | length)

    $all
        | where {|cmd| $cmd.name | str starts-with $prefix }
        | each {|cmd|
            let parts = ($cmd.name | split row " ")
            {
                value: ($parts | get $position)
                description: (
                    if ($parts | length) == ($position + 1) { $cmd.description } else { "…" }
                )
            }
        }
        | uniq-by value
}

# Show the encoded queue map.
export def partitions [] { $PARTITIONS }

# Your queued/running jobs.
export def queue [] { squeue --me --format "%.18i %.30j %.18P %.24q %.10T %.19S %.10M %R" }

# Submit an ironmc target as a slurm job.
export def submit [
    ...target: string@target-completions  # e.g.: ladder run 50 — or quoted if it contains flags: "ladder guard-tau --baseline -0.25"
    --time: string = "04:00:00"    # walltime HH:mm:ss (capped at 10 min with --devel)
    --partition: string = "zen3_0512"
    --qos: string              # defaults to the partition's default QoS
    --devel                    # devel QoS: fixed 10-minute walltime
    --cpus: int = 128          # zen3 nodes have 128 cores
    --job-name: string         # defaults to the target with dashes
    --dry-run                  # print the batch script instead of submitting
] {
    let target = ($target | str join " ")
    if ($target | is-empty) {
        error make {msg: "no target given; try `ironmc slurm submit ladder run 20`"}
    }
    let row = ($PARTITIONS | where partition == $partition)
    if ($row | is-empty) {
        error make {msg: $"unknown partition ($partition); see `ironmc slurm partitions`"}
    }
    let row = ($row | first)

    let qos = if $devel {
        if $row.devel_qos == null {
            error make {msg: $"partition ($partition) has no devel QoS (only zen3_0512 and zen3_0512_a100x2 do)"}
        }
        $row.devel_qos
    } else if $qos == null { $row.default_qos } else { $qos }

    if not ($time =~ '^\d{1,3}:\d{2}:\d{2}$') {
        error make {msg: $"walltime ($time) is not HH:mm:ss"}
    }
    let parts = ($time | split row ":" | each {|part| $part | into int })
    let requested_secs = ($parts.0 * 3600 + $parts.1 * 60 + $parts.2)
    # The devel QoS caps walltime at 10 minutes; a shorter --time is honored.
    let capped = ($devel and $requested_secs > 600)
    let walltime_secs = if $capped { 600 } else { $requested_secs }
    let time = if $capped { $DEVEL_WALLTIME } else { $time }

    let name = ($job_name | default ($target | str replace --all " " "-"))
    let stamp = (date now | format date "%Y%m%d-%H%M%S")
    let root = ($env.DATA | path join ironmc)
    let logdir = ($root | path join slurm)
    mkdir $logdir
    let script = ($logdir | path join $"($name)-($stamp).sh")
    let modpath = ($root | path join nu ironmc)
    let nu_exe = $nu.current-exe
    # Resolve cargo NOW and embed its directory: the batch shell on the compute
    # node does not reliably inherit the login PATH (fail here, not in-queue).
    let cargo_dir = (which cargo | get -o 0.path | default "" | path dirname)
    if $cargo_dir == "" {
        error make {msg: "cargo not found in PATH at submit time — load/install the rust toolchain first"}
    }

    let lines = [
        "#!/bin/bash"
        $"#SBATCH --job-name=($name)"
        $"#SBATCH --partition=($partition)"
        $"#SBATCH --qos=($qos)"
        $"#SBATCH --time=($time)"
        "#SBATCH --nodes=1"
        "#SBATCH --ntasks=1"
        $"#SBATCH --cpus-per-task=($cpus)"
        $"#SBATCH --output=($logdir)/%x-%j.out"
        ""
        $"export PATH=\"($cargo_dir):$HOME/.cargo/bin:$PATH\""
        "# Explicit rayon pool size; falls back to cgroup autodetection otherwise."
        "export RAYON_NUM_THREADS=\"${SLURM_CPUS_PER_TASK:-128}\""
        "# VSC-5's slurm does not export the job end time; synthesize it from"
        "# start time + walltime so the in-run deadline stop arms either way."
        "echo \"slurm-provided SLURM_JOB_END_TIME: ${SLURM_JOB_END_TIME:-unset}\""
        ("export SLURM_JOB_END_TIME=\"${SLURM_JOB_END_TIME:-$(($(date +%s) + " + ($walltime_secs | into string) + "))}\"")
        $"RESULT=\"($logdir)/($name)-${SLURM_JOB_ID}.json\""
        $"'($nu_exe)' -c 'use ($modpath); ironmc ($target) | to json out> '\"$RESULT\""
        "echo; echo '=== result ==='; cat \"$RESULT\""
    ]
    $lines | str join "\n" | save -f $script

    if $dry_run {
        print (open --raw $script)
        return {script: $script, submitted: false}
    }
    # Build everything the target might run, so `cargo run` inside the job is
    # a no-op exec instead of eating walltime (a failed build aborts here,
    # before anything is queued).
    cd $ROOT
    cargo build --release --workspace
    let out = (sbatch $script | str trim)
    print $out
    let id = ($out | parse "Submitted batch job {id}" | get -o 0.id)
    # slurm copied the script at submission, so renaming it to the job id
    # (matching the %x-%j .out and the job's $SLURM_JOB_ID result) is safe.
    let script = if $id != null {
        let renamed = ($logdir | path join $"($name)-($id).sh")
        mv $script $renamed
        $renamed
    } else { $script }
    {
        script: $script
        result: (if $id != null { $logdir | path join $"($name)-($id).json" } else { null })
        job_id: $id
        partition: $partition
        qos: $qos
        time: $time
        submitted: true
    }
}
