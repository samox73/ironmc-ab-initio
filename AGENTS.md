# Agent Instructions

## Repo Map

- `crates/rmc-core` - engine: updates, kernels, update sets, runner, seeding.
- `crates/rmc-{stats,grids,numeric,io,diagmc}` - opt-in batteries.
- `crates/rmc` - facade crate, feature-gated re-exports.
- `crates/apps/rmc-minimal` - hot-path benchmark fixture.
- `crates/apps/rmc-frohlich` - full polaron engine and perf fixture.
- `crates/apps/rmc-ab-initio` - ab-initio LiF engine (CPU slotmap + GPU); configs in `configs/`.
- `nu/ironmc` - nushell CLI: run targets (`cpu`/`gpu run`) and VSC-5 slurm dispatch (`slurm submit`).
- `datasets/` - vertex tables and other versioned inputs.
- `runs/` - experiment dirs from `slurm submit --out-dir runs/<experiment>` (required on every submission): `jobs/<stamp>-<name>/` (scripts + logs, one per submission) beside `results/` (the run's cumulative dumps; resubmitting the same --out-dir resumes the checkpointed run). Run targets read the run dir from `$IRONMC_OUT_DIR`, which submit exports. `slurm/` is the read-only legacy tree from before `--out-dir` was mandatory.

## Build & Test

- Never run the full workspace test suite yourself. When full-suite verification is needed, give the user the exact `rtk cargo test --release --workspace` command and wait for their results.
- The debug profile is off limits by default. Run Cargo builds, checks, tests, and applications with `--release` (or `--profile release-tuned` for tuned runs). Use debug only after a concrete problem requires it, and state that reason first.
- Run only the smallest relevant release-mode test/check target yourself.
- `make run` does the default Frohlich-polaron run.
- The `release-tuned` profile exists for tuned performance builds; see `README.md`.

## VSC-5 devel QoS — do not underestimate it

Both devel QoS are full production nodes, only walltime-capped (10 min). Measured: one zen3_0512_devel node builds the n=65 rmcvtab table in 5m21s.

- `zen3_0512_devel`: 5 nodes, each 128 cores + 512 GB RAM.
- `zen3_0512_a100x2_devel`: 2 nodes, each 512 GB RAM + 2× A100 (4 GPUs total).

Sizing that fits comfortably on one 512 GB node is devel-viable if it finishes (or checkpoints) within 10 minutes — don't route work to the normal QoS queue just because it "sounds big".

## Shell

Prefix shell commands with `rtk`.

Examples:

```bash
rtk git status
rtk cargo test
```

## Benchmarks

Do not run benchmark commands yourself.

Benchmark commands may require `sudo` to isolate a CPU core. When benchmark verification is needed, tell the user exactly which command to run and wait for their results.

Copy-pasteable benchmark commands from `README.md`:

```bash
cargo bench-compare -p rmc-minimal --bin rmc-minimal --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' -- full 100000000
cargo bench-compare -p rmc-minimal --bin rmc-minimal --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' --rev-base HEAD -- full 100000000
cargo bench-compare -p rmc-frohlich --bin rmc-frohlich --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' -- bench fixtures/bench-frohlich.json
```

## Conventions

- Plan files live in `plans/{open,active,done}/*.md`, sorted by status: `open` = not started, `active` = partly done, `done` = implemented.
- Prefer existing `type: summary` commit subjects such as `chore:`, `refactor:`, and `fix:`.
- Document implementation chunks in journal.md and physics notes and methodology in ./latex/ab-initio-diagmc/ab-initio-diagmc.tex
- Project terminology is defined in ./latex/glossary/glossary.tex — keep it current when new terms appear.
- Do not put plan checkpoint names (A1, B3, etc.) anywhere else. Plan files are ephemeral and may not exist at a later point.
