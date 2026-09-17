# Dispatch ironmc targets as slurm jobs on VSC-5.
#
#   ironmc slurm targets                               # job catalog
#   ironmc slurm submit { ironmc ladder run 50 --node } --time 08:00:00 --out-dir runs/ladder/n50
#   ironmc slurm submit { ironmc gates dmc } --devel --out-dir runs/gates/dmc   # 10-min devel QoS
#   ironmc slurm submit { ironmc gpu run 20 --config lif-gpu-tau60 } --out-dir runs/gpu/n20
#   ironmc slurm submit { ironmc cpu run 50 } --devel --ranks 5 --out-dir runs/e0-fit/n50
#   ironmc slurm submit { ironmc tables build 50 } --dry-run --out-dir runs/tables/n50
#
# Every job is a named experiment: --out-dir runs/<experiment> is REQUIRED. Its
# scripts/logs land in <out-dir>/jobs/<stamp>-<name>, results in <out-dir>/results.
#
# The job is a closure: inside the braces the full nu toolchain applies —
# tab completion of commands AND their flags, parse-time flag/type checking.
# It is serialized via `view source` and replayed inside sbatch, so keep it
# self-contained: captured variables (`let n = 20; submit { ladder run $n }`)
# do not exist in-job. Any pipeline works: { ladder sweep --node | ladder fit }.
#
# submit prebuilds in release mode before calling sbatch (jobs share $DATA, so
# compiled artifacts carry over and the job never spends walltime compiling). On
# GPU partitions it builds only the gpu-cuda feature set — never a plain
# no-feature workspace build too, or the two would thrash cargo's cache on every
# submit. --dry-run skips the build.

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

# Partition names from the queue map, with their properties as menu hints.
def partition-completions [] {
    $PARTITIONS | each {|row|
        {
            value: $row.partition
            description: ([
                $row.description
                (if $row.devel_qos != null { "devel QoS" })
            ] | compact | str join " — ")
        }
    }
}

# Namespaces whose commands can be dispatched as jobs. Ab-initio LiF work is
# the top-level namespaces; other physics gets its own namespace (frohlich).
const NAMESPACES = [tables ladder gates test bench qe cpu gpu frohlich]

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

# Resolve the config file a job body will run with, for provenance: an explicit
# `--config <name>` in the body, else the target command's `--config` default
# read from its signature. Returns an existing path (a name maps to
# configs/<name>.json under `code_root`; a literal path is used as-is) or null
# when the target takes no config.
def config-for-body [body: string, code_root: string] {
    let explicit = ($body | parse --regex '--config\s+(?<c>\S+)' | get -o 0.c)
    let name = if $explicit != null { $explicit } else {
        let words = ($body | str replace --regex '^ironmc\s+' '' | split row " " | where {|w| $w != "" })
        let hits = (dispatchable | where {|cmd|
            let parts = ($cmd.name | split row " ")
            ($words | length) >= ($parts | length) and (($words | take ($parts | length)) == $parts)
        })
        if ($hits | is-empty) { null } else {
            $hits | sort-by {|c| $c.name | str length} | last
                | get params | where parameter_name == "config" | get -o 0.parameter_default
        }
    }
    if $name == null { return null }
    if ($name | path exists) { return $name }
    let p = ($code_root | path join crates apps rmc-ab-initio configs $"($name).json")
    if ($p | path exists) { $p } else { null }
}

# Every dispatchable target with its arguments and defaults — the catalog of
# what's worth putting inside a `submit { … }` closure.
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

# Job dirs live in two trees: the legacy flat slurm/<stamp>-*/ and, for named
# runs, runs/**/jobs/<stamp>-*/ (see `submit --out-dir`). Search both for a
# job's .out file(s). `filename` is a glob (e.g. job-123.out, *.out).
def job-log-search [filename: string] {
    [
        (slurm-logdir | path join "*" $filename)
        ($env.DATA | path join ironmc runs "**" jobs "*" $filename)
    ] | each {|p| try { ls ($p | into glob) } catch { [] } } | flatten
}

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
    let hits = (job-log-search $"job-($job_id).out")
    if ($hits | is-empty) {
        error make {msg: $"cannot resolve stdout of job ($job_id): not in scontrol, no match under (slurm-logdir) or runs/"}
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
        let logs = (job-log-search "*.out" | sort-by modified)
        if ($logs | is-empty) {
            error make {msg: $"no job id given, queue is empty, and no logs under (slurm-logdir) or runs/"}
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

# Submit a job as a slurm batch. Everything inside the closure braces is the
# job (with native tab completion and parse-time checking); submit's own flags
# shape the allocation. The closure is serialized as text and replayed in-job,
# so it must be self-contained — no captured variables.
export def submit [
    job: closure               # e.g. { ironmc ladder run 50 --node }
    --time: string = "04:00:00"    # walltime HH:mm:ss (capped at 10 min with --devel)
    --partition: string@partition-completions = "zen3_0512"
    --devel                    # devel QoS: fixed 10-minute walltime
    --cpus: int = 128          # zen3 nodes have 128 cores
    --ranks: int = 1           # independent MC ranks (each reads $SLURM_PROCID): GPU partitions pack one per GPU, CPU partitions give each a whole node
    --job-name: string         # defaults to the job text with dashes
    --commit-hash: string      # pin the job's code: build in a detached worktree of this
                               # commit; the job runs that tree's code (data paths stay in the
                               # main repo via $IRONMC_DATA_ROOT) and removes the worktree when
                               # it finishes
    --out-dir: string          # REQUIRED experiment dir under runs/ (e.g. runs/e0-fit/n20): this
                               # submission's scripts/logs land in <out-dir>/jobs/<stamp>-<name>,
                               # and the job sees $IRONMC_OUT_DIR = <out-dir>/results, which run
                               # targets take as their --out-dir default (cumulative dumps).
                               # Resubmit with the same --out-dir to resume the checkpointed run —
                               # the N submissions that build one result set sit beside it in jobs/.
    --dry-run                  # print the batch script instead of submitting
] {
    let body = (view source $job
        | str trim
        | str replace --regex '^\{\s*' ''
        | str replace --regex '\s*\}$' ''
        | str trim)
    if ($body | is-empty) {
        error make {msg: "empty job; try `ironmc slurm submit { ironmc ladder run 20 } --out-dir runs/demo`"}
    }
    if ($body | str starts-with "|") {
        error make {msg: "the job closure must not take parameters — flags go after the braces"}
    }
    # job.sh replays the body inside a single-quoted bash string; a ' in the
    # body would end that string early and run the rest as shell.
    if ($body | str contains "'") {
        error make {msg: "the job closure must not contain single quotes — nu string literals can use double quotes instead"}
    }
    if ($env.DATA? | is-empty) {
        error make {msg: "submit needs $env.DATA (the VSC-5 shared filesystem) — run it on the cluster"}
    }
    # Table builds derive their run dir from the grid in the body: `submit {
    # ironmc tables shard 80 }` (also merge-shards / build / reference) targets
    # runs/tables/n80, so a grid's shard and merge jobs share one dir and the
    # merge finds the shards. An explicit --out-dir always wins.
    let out_dir = if $out_dir != null { $out_dir } else {
        let m = ($body | parse --regex 'tables\s+(?:merge-shards|shard|build|reference)\s+(?<n>\d+)')
        if ($m | is-not-empty) { $"runs/tables/n($m.0.n)" } else { null }
    }
    # Every job is a named experiment under runs/: scripts/logs go to
    # <out-dir>/jobs/, results to <out-dir>/results/. No anonymous jobs.
    if $out_dir == null {
        error make {msg: "every job needs --out-dir runs/<experiment> (table builds derive it from the grid, e.g. `tables shard 80` → runs/tables/n80)"}
    }
    # Constrained under runs/ so job-log-search and the Makefile sync/mount
    # targets find everything by construction; the charset keeps the path safe
    # to embed unquoted in job.sh and #SBATCH directives.
    if (not ($out_dir =~ '^runs/[A-Za-z0-9._/-]+$')) or ($out_dir =~ '\.\.') {
        error make {msg: $"--out-dir must be a relative dir under runs/, using only [A-Za-z0-9._/-] — e.g. runs/e0-fit/n50; got: ($out_dir)"}
    }
    if ($body =~ '--out-dir[=\s]+\S+') {
        error make {msg: "--out-dir is a `slurm submit` flag — drop it from the job closure"}
    }
    let row = ($PARTITIONS | where partition == $partition)
    if ($row | is-empty) {
        error make {msg: $"unknown partition ($partition); see `ironmc slurm partitions`"}
    }
    let row = ($row | first)

    # VSC-5 QoS is partition-determined (default QoS = partition name), so
    # there is no separate --qos knob; --devel picks the devel variant.
    let qos = if $devel {
        if $row.devel_qos == null {
            error make {msg: $"partition ($partition) has no devel QoS — only zen3_0512 and zen3_0512_a100x2 do"}
        }
        $row.devel_qos
    } else { $row.default_qos }

    if not ($time =~ '^\d{1,3}:\d{2}:\d{2}$') {
        error make {msg: $"walltime ($time) is not HH:mm:ss"}
    }
    let parts = ($time | split row ":" | each {|part| $part | into int })
    let requested_secs = ($parts.0 * 3600 + $parts.1 * 60 + $parts.2)
    # The devel QoS caps walltime at 10 minutes; a shorter --time is honored.
    let capped = ($devel and $requested_secs > 600)
    let walltime_secs = if $capped { 600 } else { $requested_secs }
    let time = if $capped { $DEVEL_WALLTIME } else { $time }

    let name = ($job_name | default (
        $body | str replace --regex '^ironmc\s+' ''
              | str replace --all --regex '[^A-Za-z0-9._]+' '-'
              | str trim --char "-"
    ))
    # Fold the experiment into the default name: distinct experiments submitting
    # the same closure must stay tellable-apart in squeue/history/completions,
    # and the still-queued guard below matches jobs by name.
    let name = if $job_name == null {
        $"($name)-($out_dir | str replace --regex '^runs/' '' | str replace --all --regex '[^A-Za-z0-9._]+' '-')"
    } else { $name }
    let stamp = (date now | format date "%Y%m%d-%H%M%S")
    let root = ($env.DATA | path join ironmc)
    # Run targets (cpu/gpu run|flat) accumulate dumps in the run dir and resume
    # from a checkpoint there; every other target just returns a value. The run
    # dir reaches the job as $IRONMC_OUT_DIR (run targets read it as their
    # --out-dir default) — never by editing the closure text, so any body shape
    # works. ponytail: run-ness is sniffed from the body text — a new run-like
    # target must be added to this regex; misclassification is benign
    # ($IRONMC_OUT_DIR is exported regardless; only merge/stdout capture differ).
    let is_run = ($body =~ '\b(gpu|cpu) (run|flat)\b')
    # One experiment dir holds jobs/ (this and every resubmission's scripts/logs)
    # beside results/ (the single dump set they build), so the 1-to-N job→result
    # relationship is explicit and co-located. The stamp prefix makes
    # lexicographic order = submission order = job-id order (ids are monotone in
    # time), and the dir name is known before sbatch, so scripts, output, and
    # results are born in their final place.
    let jobdir = ($root | path join $out_dir jobs $"($stamp)-($name)")
    # Where a run target's cumulative dumps land (absolute): exported as
    # $IRONMC_OUT_DIR and merged all-ranks/all-legs at the end of the job.
    let run_dir = ($root | path join $out_dir results)
    let script = ($jobdir | path join "job.sh")
    # Pinned jobs run the code of a detached worktree (unique per submission,
    # so cleanup by one job can never yank another's tree); data paths keep
    # resolving into the main repo through $IRONMC_DATA_ROOT.
    let pinned = if $commit_hash != null {
        let hash = (git -C $root rev-parse --verify --short=9 $commit_hash | str trim)
        $root | path join ".worktrees" $"($hash)-($stamp)"
    } else { null }
    let code_root = ($pinned | default $root)
    let modpath = ($code_root | path join nu ironmc)
    let nu_exe = $nu.current-exe
    # Resolve cargo NOW and embed its directory: the batch shell on the compute
    # node does not reliably inherit the login PATH (fail here, not in-queue).
    let cargo_dir = (which cargo | get -o 0.path | default "" | path dirname)
    if $cargo_dir == "" {
        error make {msg: "cargo not found in PATH at submit time — load/install the rust toolchain first"}
    }

    # --ranks = independent MC ranks (each its own $SLURM_PROCID + RankDump). A
    # GPU partition packs ranks onto GPUs (≤ gpus/node, or a multiple → several
    # nodes); a CPU partition gives each rank a whole node (1 rank/node, all
    # cores) — this is how the devel QoS's 5 zen3 nodes become one 640-core run.
    if $ranks < 1 {
        error make {msg: "--ranks must be ≥ 1"}
    }
    if $row.gpus > 0 and $ranks > $row.gpus and $ranks mod $row.gpus != 0 {
        error make {msg: $"($partition) has ($row.gpus) GPUs per node; --ranks must be ≤ ($row.gpus) or a multiple of it"}
    }
    let nodes = if $row.gpus > 0 {
        if $ranks > $row.gpus { $ranks // $row.gpus } else { 1 }
    } else {
        $ranks
    }
    let ranks_per_node = if $ranks > 1 { ($ranks // $nodes) } else { 1 }
    let ntasks = $ranks
    let cpus_per_task = if $row.gpus > 0 and $ranks > 1 { $cpus // $ranks_per_node } else { $cpus }
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
        $"#SBATCH --nodes=($nodes)"
        $"#SBATCH --ntasks=($ntasks)"
        $"#SBATCH --cpus-per-task=($cpus_per_task)"
    ] | append $gpu_directives
      | append [
        $"#SBATCH --output=($jobdir)/job-%j.out"
        ""
        $"export PATH=\"($cargo_dir):$HOME/.cargo/bin:$PATH\""
        "# Explicit rayon pool size; falls back to cgroup autodetection otherwise."
        "export RAYON_NUM_THREADS=\"${SLURM_CPUS_PER_TASK:-128}\""
        $"export IRONMC_DATA_ROOT=\"($root)\""
        "# VSC-5's slurm does not export the job end time; synthesize it from"
        "# start time + walltime so the in-run deadline stop arms either way."
        "echo \"slurm-provided SLURM_JOB_END_TIME: ${SLURM_JOB_END_TIME:-unset}\""
        ("export SLURM_JOB_END_TIME=\"${SLURM_JOB_END_TIME:-$(($(date +%s) + " + ($walltime_secs | into string) + "))}\"")
        $"export IRONMC_OUT_DIR=\"($run_dir)\""
    ])
    # `cpu run`/`gpu run` write their own result files into the run dir, so their
    # stdout is purely a log; every other target's return value is captured into
    # the job dir (`save` picks the format from the .json extension). Import both
    # prefixed and flat so the body works however the module was loaded.
    let invoke_lines = if $is_run {
        [$"'($nu_exe)' -c 'use ($modpath); use ($modpath) *; \(($body)\)'"]
    } else {
        [
            $"RESULT=\"($jobdir)/result-${SLURM_JOB_ID}.json\""
            $"'($nu_exe)' -c 'use ($modpath); use ($modpath) *; \(($body)\) | save -f '\"$RESULT\""
            "echo; echo '=== result ==='; cat \"$RESULT\""
        ]
    }
    let lines = ($lines | append $invoke_lines)
    # `run_dir` (always set — every job has an out-dir) is where a run target's
    # cumulative dumps land; merge them all-ranks/all-legs at the end of the job.
    # For non-run targets the compgen guard makes this a no-op.
    let merge_lines = [
        $"DUMPS=\"($run_dir)\""
        "if compgen -G \"$DUMPS/batches-rank*.bin\" > /dev/null; then"
        $"    '($code_root)/target/release/rmc-ab-initio' merge --out \"$DUMPS/merged.json\" \"$DUMPS\"/batches-rank*.bin"
        "    echo; echo '=== merged report (all ranks, all legs) ==='; cat \"$DUMPS/merged.json\""
        "fi"
    ]
    # Multi-rank jobs: each srun task runs the target with its own $SLURM_PROCID
    # (rank) and result file. GPU tasks bind one device each; CPU tasks take a
    # whole node. Rank-aware targets (cpu run / gpu run) read $SLURM_PROCID.
    let task_script = ($jobdir | path join "task.sh")
    let task_text = if $ranks > 1 {
        let cuda_bind = if $row.gpus > 0 {
            [
                "# One GPU per task: bind by node-local task id unless slurm already"
                "# restricted this task to a single device."
                "if [ -z \"${CUDA_VISIBLE_DEVICES:-}\" ] || [[ \"$CUDA_VISIBLE_DEVICES\" == *,* ]]; then"
                "    export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}"
                "fi"
            ]
        } else { [] }
        let task_body = if $is_run {
            [$"'($nu_exe)' -c 'use ($modpath); use ($modpath) *; \(($body)\)'"]
        } else {
            [
                $"RESULT=\"($jobdir)/result-${SLURM_JOB_ID}-rank${SLURM_PROCID}.json\""
                $"'($nu_exe)' -c 'use ($modpath); use ($modpath) *; \(($body)\) | save -f '\"$RESULT\""
                "echo; echo \"=== result rank ${SLURM_PROCID} ===\"; cat \"$RESULT\""
            ]
        }
        # Print each task's actual core count — the first thing to check if a CPU
        # multi-rank run is slow (rayon can only use what srun binds to the task).
        let diag = ["echo \"rank ${SLURM_PROCID}: nproc=$(nproc) cpus-per-task=${SLURM_CPUS_PER_TASK} rayon=${RAYON_NUM_THREADS}\""]
        (["#!/bin/bash"] | append $cuda_bind | append $diag | append $task_body | str join "\n")
    } else { null }
    let lines = if $ranks > 1 {
        # srun (slurm ≥ 22.05) no longer inherits --cpus-per-task from the batch
        # allocation, so each task would be pinned to a single CPU and rayon would
        # run on a ~1-core cpuset (measured: ~25× slowdown). Pass it explicitly so
        # every task gets its full node's cores.
        ($lines | drop ($invoke_lines | length) | append [$"srun --cpus-per-task=($cpus_per_task) bash '($task_script)'"] | append $merge_lines)
    } else {
        ($lines | append $merge_lines)
    }
    # A pinned job removes its private worktree when it finishes, so target/
    # dirs don't accumulate (a killed job leaves one; `git worktree remove`).
    let lines = if $pinned != null {
        ($lines | append [$"git -C '($root)' worktree remove --force '($pinned)' || true"])
    } else { $lines }
    let script_text = ($lines | str join "\n")

    if $dry_run {
        # Nothing is written on a dry run — no stray <stamp> dirs in the tree.
        print $script_text
        if $task_text != null { print $"\n# --- task.sh ---\n($task_text)" }
        return {script: null, submitted: false}
    }
    # A second job writing the same <out-dir>/results mid-run corrupts the
    # checkpointed resume, so refuse while any job of this experiment is still
    # in the queue. Default names end in the folded experiment dir, so matching
    # that suffix also catches a *different* body into the same dir (e.g.
    # merge-shards submitted while the shard legs are still running). Explicit
    # --job-name jobs are only caught on an exact name match.
    if (which squeue | is-not-empty) {
        # --Format with an explicit width: the default %j column truncates
        # long names, and ours carry the whole body + experiment path.
        let queued = (squeue --me --noheader --Format "Name:256" | lines | str trim)
        let exp = ($out_dir | str replace --regex '^runs/' ''
            | str replace --all --regex '[^A-Za-z0-9._]+' '-')
        let clash = ($queued | where {|q| $q == $name or ($q | str ends-with $"-($exp)") })
        if ($clash | is-not-empty) {
            error make {msg: $"($clash | first) is already queued/running — wait or `ironmc slurm cancel` it before submitting another job into ($run_dir)"}
        }
    }
    mkdir $jobdir
    $script_text | save -f $script
    if $task_text != null { $task_text | save -f $task_script }
    # Build everything the target might run, so `cargo run` inside the job is
    # a no-op exec instead of eating walltime (a failed build aborts here,
    # before anything is queued). A pinned job builds in its own detached
    # worktree; the main checkout is untouched.
    if $pinned != null {
        git -C $root worktree add --detach $pinned $commit_hash
    }
    cd $code_root
    # Copy the run's config into the job dir for provenance: job.sh records the
    # target text, but not the config's *contents* (which may change later). Read
    # from $code_root so a pinned job captures that commit's config.
    let cfg_src = (config-for-body $body $code_root)
    if $cfg_src != null {
        cp $cfg_src ($jobdir | path join ($cfg_src | path basename))
    }
    if $row.gpus > 0 {
        # GPU jobs run the gpu-cuda binary (`cargo run --features gpu-cuda`), so
        # build exactly that feature set — and ONLY it. A plain `--workspace`
        # build compiles cubecl/rmc-ab-initio *without* gpu-cuda; the lines below
        # then build them *with* it. Cargo keeps one feature variant per target
        # dir, so having both toggles the `cubecl/cuda` feature off then on and
        # recompiles cubecl + rmc-ab-initio + rmc-frohlich on every submit.
        # Building only the gpu-cuda set keeps the cache warm across submits.
        # cudarc pins the CUDA version via CUDA_PATH at build time; match the
        # toolkit the in-job `ironmc cuda load` will point at.
        cuda load
        cargo build --release -p rmc-ab-initio --features gpu-cuda
        cargo build --release -p rmc-frohlich --features gpu-cuda
        # `gpu parity` runs cargo test in-job; precompile the test binaries.
        cargo test --release -p rmc-ab-initio --features gpu-cuda --no-run
    } else {
        cargo build --release --workspace
    }
    let out = (sbatch $script | str trim)
    print $out
    let id = ($out | parse "Submitted batch job {id}" | get -o 0.id)
    {
        jobdir: $jobdir
        run_dir: $run_dir
        # Only single-rank non-run targets save a return value; a run target's
        # data is run_dir, multi-rank results are result-<id>-rank<r>.json.
        result: (if $id != null and not $is_run and $ranks == 1 { $jobdir | path join $"result-($id).json" } else { null })
        job_id: $id
        partition: $partition
        qos: $qos
        time: $time
        submitted: true
    }
}
