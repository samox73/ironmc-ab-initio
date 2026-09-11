# Agent Instructions

## Repo Map

- `crates/rmc-core` - engine: updates, kernels, update sets, runner, seeding.
- `crates/rmc-{stats,grids,numeric,io}` - opt-in batteries.
- `crates/rmc` - facade crate, feature-gated re-exports.
- `crates/apps/rmc-minimal` - hot-path benchmark fixture.
- `crates/apps/rmc-frohlich` - full polaron engine and perf fixture.

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
