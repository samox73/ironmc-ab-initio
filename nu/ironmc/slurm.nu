# Dispatch ironmc targets as slurm jobs on VSC-5.
#
#   ironmc slurm targets                               # what can be submitted
#   ironmc slurm submit ladder run 50 --time 08:00:00
#   ironmc slurm submit gates dmc --devel              # 10-min devel QoS
#   ironmc slurm submit gpu run 20 --config ladder-lif-gpu
#   ironmc slurm submit tables build 50 --dry-run      # inspect, don't submit
#
# submit is --wrapped: the flags it declares itself (--time, --partition, …)
# are its own; every other flag passes through to the target unquoted. Tab
# after a full target name completes the target's flags.
#
# submit prebuilds the whole workspace in release mode before calling sbatch
# (jobs share $DATA, so compiled artifacts carry over and the job never spends
# walltime compiling). --dry-run skips the build.

use common.nu [ROOT]
use cuda.nu

# VSC-5 queue map (gitlab.tuwien.ac.at vsc5_queues.md, fetched 2026-09-07).
export const PARTITIONS = [
    [partition, default_qos, devel_qos, max_walltime, gpus, description];
    [zen3_0512, zen3_0512, zen3_0512_devel, "72:00:00", 0, "default, 512 GB"]
    [zen3_1024, zen3_1024, null, "72:00:00", 0, "high memory, 1 TB"]
    [zen3_2048, zen3_2048, null, "72:00:00", 0, "higher memory, 2 TB"]
    [zen2_0256_a40x2, zen2_0256_a40x2, null, "72:00:00", 2, "GPU, 2× A40"]
    [zen3_0512_a100x2, zen3_0512_a100x2, zen3_0512_a100x2_devel, "72:00:00", 2, "GPU, 2× A100"]
]

const DEVEL_WALLTIME = "00:10:00"

# Namespaces whose commands can be dispatched as jobs. Ab-initio LiF work is
# the top-level namespaces; other physics gets its own namespace (frohlich).
const NAMESPACES = [tables ladder gates test bench qe gpu frohlich]

# Dispatchable targets, discovered from scope with their signatures, so the
# list stays current automatically.
def dispatchable [] {
    let pattern = ('^(ironmc )?(' + ($NAMESPACES | str join "|") + ') ')
    scope commands
        | where name =~ $pattern
        | each {|cmd|
            {
                name: ($cmd.name | str replace "ironmc " "")
                description: $cmd.description
                params: ($cmd.signatures | values | get -o 0 | default []
                    | where {|p| $p.parameter_type not-in ["input", "output"] })
            }
        }
        | uniq-by name
}

# The dispatchable command whose name is a word-prefix of `words` (longest
# wins), or null. `words` may continue past the name with positionals/flags.
def resolve-target [words: list<string>] {
    let hits = (dispatchable
        | where {|cmd|
            let parts = ($cmd.name | split row " ")
            ($words | length) >= ($parts | length) and (($words | take ($parts | length)) == $parts)
        }
        | sort-by name)
    if ($hits | is-empty) { null } else { $hits | last }
}

# Context-aware completer: completes the target name ONE word at a time (some
# nu builds insert multi-word completion values unquoted), and once a full
# name is committed, completes that target's own flags (submit is --wrapped,
# so they pass through unquoted).
def target-completions [context: string] {
    # Words already committed after `submit` (the in-progress token is
    # dropped — nu filters candidates by it).
    let tail = ($context | str replace --regex '.*\bsubmit\s*' "")
    mut words = ($tail | split row " " | where {|word| $word != "" })
    if not ($context | str ends-with " ") {
        $words = ($words | drop 1)
    }
    let words = $words
    let name_words = ($words | take while {|word| not ($word | str starts-with "-") })

    let target = (resolve-target $name_words)
    if $target != null {
        let used = ($words | each {|word| $word | split row "=" | first })
        return ($target.params
            | where {|p| $p.parameter_type in ["named", "switch"] }
            | where {|p| not ($"--($p.parameter_name)" in $used) }
            | each {|p|
                {
                    value: $"--($p.parameter_name)"
                    description: ([
                        (if $p.parameter_default != null { $"default: ($p.parameter_default)" })
                        (if ($p.description | default "" | is-not-empty) { $p.description })
                    ] | compact | str join " — ")
                }
            })
    }

    let prefix = if ($name_words | is-empty) { "" } else { ($name_words | str join " ") + " " }
    let position = ($name_words | length)
    dispatchable
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

# Every dispatchable target with its arguments and defaults — what `submit`
# accepts before its own flags take over.
export def targets [] {
    dispatchable | each {|cmd|
        {
            target: $cmd.name
            args: ($cmd.params | each {|p|
                let default = (if $p.parameter_default != null { $"=($p.parameter_default)" } else { "" })
                match $p.parameter_type {
                    "positional" => $"<($p.parameter_name)($default)>"
                    "rest" => $"<($p.parameter_name)...>"
                    "switch" => $"--($p.parameter_name)"
                    "named" => $"--($p.parameter_name)($default)"
                    _ => ""
                }
            } | str join " ")
            description: $cmd.description
        }
    }
}

# Show the encoded queue map.
export def partitions [] { $PARTITIONS }

# Your queued/running jobs.
export def queue [] { squeue --me --format "%.18i %.30j %.18P %.24q %.10T %.19S %.10M %R" }

# Finished jobs from slurm accounting, newest first (running/pending live in
# `queue`). --since takes sacct time syntax: now-2days, 2026-09-01, …
export def history [
    --since: string = "now-7days"
    --all (-a)          # include still-running/pending jobs too
] {
    let rows = (sacct -X --noheader --parsable2 --starttime $since
        --format JobID,JobName,Partition,State,Elapsed,End,ExitCode
        | lines
        | where {|line| $line != "" }
        | each {|line|
            let cols = ($line | split row "|")
            {
                job_id: ($cols | get 0)
                name: ($cols | get 1)
                partition: ($cols | get 2)
                state: ($cols | get 3)
                elapsed: ($cols | get 4)
                end: ($cols | get 5)
                exit_code: ($cols | get 6)
            }
        })
    let rows = if $all { $rows } else {
        $rows | where {|row| not ($row.state in ["RUNNING", "PENDING"]) }
    }
    # sacct returns ascending job order; newest on top reads better.
    $rows | reverse
}

def slurm-logdir [] { $env.DATA | path join ironmc slurm }

# Newest job id in your queue (by submission time), or null when nothing is queued.
def newest-job-id [] {
    if (which squeue | is-empty) { return null }
    let ids = (squeue --me --noheader --sort=-V --format "%i" | lines | str trim | where {|l| $l != "" })
    if ($ids | is-empty) { null } else { $ids | first }
}

# A job's stdout path: from scontrol while slurm still knows the job, else from
# the submit log dir (`<name>-<jobid>.out`).
def stdout-path [job_id: string] {
    let info = (try { do { scontrol show job $job_id } | complete } catch { {exit_code: 127, stdout: ""} })
    if $info.exit_code == 0 {
        let parsed = ($info.stdout | parse -r 'StdOut=(?<path>\S+)')
        if ($parsed | is-not-empty) { return ($parsed | first | get path) }
    }
    let hits = (try { ls (slurm-logdir | path join "*" $"job-($job_id).out" | into glob) } catch { [] })
    if ($hits | is-empty) {
        error make {msg: $"cannot resolve stdout of job ($job_id): not in scontrol, no match in (slurm-logdir)"}
    }
    $hits | first | get name
}

# Completion menu for job ids: `<id>   <name> [<state>]`, newest first; the
# name/state column is a visual guide — selecting inserts only the id.
def job-id-completions [] {
    if (which squeue | is-empty) { return [] }
    squeue --me --noheader --sort=-V --format "%i\t%j\t%T"
        | lines
        | str trim
        | where {|line| $line != "" }
        | each {|line|
            let cols = ($line | split row "\t")
            {
                value: ($cols | get 0)
                description: $"($cols | get -o 1 | default '')  [($cols | get -o 2 | default '?')]"
            }
        }
}

# Print a job's stdout log. Defaults to your newest queued/running job (or, with
# an empty queue, the newest log file). --tail keeps watching (Ctrl-C to stop);
# it also waits for the file when the job is still pending.
export def logs [
    job_id?: string@job-id-completions  # slurm job id; any job, not only `submit`-made ones
    --tail (-t)             # follow live via tail -F instead of a one-shot cat
    --lines (-n): int = 50  # trailing lines to show when following
] {
    let id = ($job_id | default (newest-job-id))
    let path = if $id != null {
        stdout-path $id
    } else {
        let logs = (try { ls (slurm-logdir | path join "*" "*.out" | into glob) | sort-by modified } catch { [] })
        if ($logs | is-empty) {
            error make {msg: $"no job id given, queue is empty, and no logs in (slurm-logdir)"}
        }
        $logs | last | get name
    }

    if $tail {
        print $"following ($path) — Ctrl-C to stop"
        tail -n $lines -F $path
    } else if ($path | path exists) {
        open --raw $path
    } else {
        print $"($path) does not exist yet — job still pending? try --tail to wait for it"
    }
}

# Cancel queued/running jobs. Requires explicit ids (no "newest" default —
# cancelling shouldn't guess); tab completes them from your queue.
export def cancel [
    ...job_ids: string@job-id-completions  # one or more job ids
] {
    if ($job_ids | is-empty) {
        error make {msg: "no job id given; try `ironmc slurm cancel <tab>`"}
    }
    scancel ...$job_ids
    print $"cancelled: ($job_ids | str join ', ')"
    queue
}

# Submit an ironmc target as a slurm job. --wrapped: flags not declared below
# pass through to the target (e.g. submit gpu run 20 --config ladder-lif-gpu).
export def --wrapped submit [
    ...target: string@target-completions  # e.g.: ladder run 50 --node
    --time: string = "04:00:00"    # walltime HH:mm:ss (capped at 10 min with --devel)
    --partition: string = "zen3_0512"
    --qos: string              # defaults to the partition's default QoS
    --devel                    # devel QoS: fixed 10-minute walltime
    --cpus: int = 128          # zen3 nodes have 128 cores
    --gpu-ranks: int = 1       # >1: srun that many tasks, one GPU each; the target reads $SLURM_PROCID (gpu run does)
    --job-name: string         # defaults to the target with dashes
    --dry-run                  # print the batch script instead of submitting
] {
    # --wrapped disables nu's automatic --help (flags must pass through to the
    # target); answer it from the catalog instead of submitting.
    if ($target | any {|word| $word in ["--help", "-h"] }) {
        let prefix = ($target | take while {|word| not ($word | str starts-with "-") } | str join " ")
        return (targets | where {|row| $row.target | str starts-with $prefix })
    }
    if ($target | is-empty) {
        error make {msg: "no target given; try `ironmc slurm submit ladder run 20`"}
    }
    if ($env.DATA? | is-empty) {
        error make {msg: "submit needs $env.DATA (the VSC-5 shared filesystem) — run it on the cluster"}
    }
    # Fail fast on flags the target does not accept — a typo'd flag would
    # otherwise ride into the queue and die only at runtime.
    let resolved = (resolve-target $target)
    if $resolved != null {
        let accepted = ($resolved.params
            | where {|p| $p.parameter_type in ["named", "switch"] }
            | each {|p| $"--($p.parameter_name)" })
        let bad = ($target
            | where {|word| $word | str starts-with "--" }
            | each {|word| $word | split row "=" | first }
            | where {|flag| not ($flag in $accepted) })
        if ($bad | is-not-empty) {
            error make {msg: $"($resolved.name) does not take ($bad | str join ', '); see `ironmc slurm targets`"}
        }
    }
    let target = ($target | str join " ")
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
    # One directory per job, prefixed by the submission stamp: lexicographic
    # listing/tab order = submission order = job-id order (ids are monotone in
    # time), and — unlike a job-id prefix — the name is known before sbatch,
    # so scripts, output, and results are born in their final place.
    let jobdir = ($env.DATA | path join ironmc slurm $"($stamp)-($name)")
    mkdir $jobdir
    let script = ($jobdir | path join "job.sh")
    let modpath = ($root | path join nu ironmc)
    let nu_exe = $nu.current-exe
    # Resolve cargo NOW and embed its directory: the batch shell on the compute
    # node does not reliably inherit the login PATH (fail here, not in-queue).
    let cargo_dir = (which cargo | get -o 0.path | default "" | path dirname)
    if $cargo_dir == "" {
        error make {msg: "cargo not found in PATH at submit time — load/install the rust toolchain first"}
    }

    if $gpu_ranks > $row.gpus and $gpu_ranks > 1 {
        error make {msg: $"($partition) has ($row.gpus) GPUs per node; --gpu-ranks ($gpu_ranks) does not fit"}
    }
    let ntasks = if $gpu_ranks > 1 { $gpu_ranks } else { 1 }
    let cpus_per_task = if $gpu_ranks > 1 { $cpus // $gpu_ranks } else { $cpus }
    # The GPU count is a per-partition constant (queue map above). VSC-5's
    # submit filter would default to all node GPUs anyway, but it only parses
    # --gres=gpu:N — never use --gpus-per-task (jobs pend forever); the
    # per-task device split happens via CUDA_VISIBLE_DEVICES in the task
    # script instead.
    let gpu_directives = if $row.gpus > 0 {
        [$"#SBATCH --gres=gpu:($row.gpus)"]
    } else {
        []
    }
    let lines = ([
        "#!/bin/bash"
        $"#SBATCH --job-name=($name)"
        $"#SBATCH --partition=($partition)"
        $"#SBATCH --qos=($qos)"
        $"#SBATCH --time=($time)"
        "#SBATCH --nodes=1"
        $"#SBATCH --ntasks=($ntasks)"
        $"#SBATCH --cpus-per-task=($cpus_per_task)"
    ] | append $gpu_directives
      | append [
        $"#SBATCH --output=($jobdir)/job-%j.out"
        ""
        $"export PATH=\"($cargo_dir):$HOME/.cargo/bin:$PATH\""
        "# Explicit rayon pool size; falls back to cgroup autodetection otherwise."
        "export RAYON_NUM_THREADS=\"${SLURM_CPUS_PER_TASK:-128}\""
        "# VSC-5's slurm does not export the job end time; synthesize it from"
        "# start time + walltime so the in-run deadline stop arms either way."
        "echo \"slurm-provided SLURM_JOB_END_TIME: ${SLURM_JOB_END_TIME:-unset}\""
        ("export SLURM_JOB_END_TIME=\"${SLURM_JOB_END_TIME:-$(($(date +%s) + " + ($walltime_secs | into string) + "))}\"")
        $"RESULT=\"($jobdir)/result-${SLURM_JOB_ID}.json\""
        # `save` picks the format from the extension: structured results become
        # real JSON, plain-text targets (external stdout) are written raw
        # instead of one escaped JSON string full of literal \n.
        $"'($nu_exe)' -c 'use ($modpath); ironmc ($target) | save -f '\"$RESULT\""
        "echo; echo '=== result ==='; cat \"$RESULT\""
    ])
    # Multi-rank GPU jobs: each srun task runs the target with its own GPU and
    # result file; rank-aware targets (gpu run) read $SLURM_PROCID themselves.
    let lines = if $gpu_ranks > 1 {
        let task_script = ($jobdir | path join "task.sh")
        [
            "#!/bin/bash"
            "# One GPU per task: bind by node-local task id unless slurm already"
            "# restricted this task to a single device."
            "if [ -z \"${CUDA_VISIBLE_DEVICES:-}\" ] || [[ \"$CUDA_VISIBLE_DEVICES\" == *,* ]]; then"
            "    export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}"
            "fi"
            $"RESULT=\"($jobdir)/result-${SLURM_JOB_ID}-rank${SLURM_PROCID}.json\""
            $"'($nu_exe)' -c 'use ($modpath); ironmc ($target) | save -f '\"$RESULT\""
            "echo; echo \"=== result rank ${SLURM_PROCID} \(GPU $CUDA_VISIBLE_DEVICES\) ===\"; cat \"$RESULT\""
        ] | str join "\n" | save -f $task_script
        ($lines | drop 3 | append [$"srun bash '($task_script)'"])
    } else {
        $lines
    }
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
    if $row.gpus > 0 {
        # cudarc pins the CUDA version via CUDA_PATH at build time; match the
        # toolkit the in-job `ironmc cuda load` will point at.
        cuda load
        cargo build --release -p rmc-ab-initio --features gpu-cuda
        cargo build --release -p rmc-frohlich --features gpu-cuda
        # `gpu parity` runs cargo test in-job; precompile the test binaries.
        cargo test --release -p rmc-ab-initio --features gpu-cuda --no-run
    }
    let out = (sbatch $script | str trim)
    print $out
    let id = ($out | parse "Submitted batch job {id}" | get -o 0.id)
    {
        jobdir: $jobdir
        result: (if $id != null { $jobdir | path join $"result-($id).json" } else { null })
        job_id: $id
        partition: $partition
        qos: $qos
        time: $time
        submitted: true
    }
}
