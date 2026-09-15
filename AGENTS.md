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
- `runs/` - experiment dirs from `slurm submit --out-dir runs/<experiment>`: `jobs/<stamp>-<name>/` (scripts + logs, one per submission) beside `results/` (the run's cumulative dumps; resubmitting the same --out-dir resumes the checkpointed run). Run targets read the run dir from `$IRONMC_OUT_DIR`, which submit exports. `slurm/` is the legacy flat job-dir tree.

## Build & Test

- Never run the full workspace test suite yourself. When full-suite verification is needed, give the user the exact `rtk cargo test --release --workspace` command and wait for their results.
- The debug profile is off limits by default. Run Cargo builds, checks, tests, and applications with `--release` (or `--profile release-tuned` for tuned runs). Use debug only after a concrete problem requires it, and state that reason first.
- Run only the smallest relevant release-mode test/check target yourself.
- `make run` does the default Frohlich-polaron run.
- The `release-tuned` profile exists for tuned performance builds; see `README.md`.

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

- Plan files live in `plans/*.md`.
- Prefer existing `type: summary` commit subjects such as `chore:`, `refactor:`, and `fix:`.
- Document implementation chunks in journal.md and physics notes and methodology in ./latex/ab-initio-diagmc/ab-initio-diagmc.tex
- Project terminology is defined in ./latex/glossary/glossary.tex — keep it current when new terms appear.
- Do not put plan checkpoint names (A1, B3, etc.) anywhere else. Plan files are ephemeral and may not exist at a later point.
