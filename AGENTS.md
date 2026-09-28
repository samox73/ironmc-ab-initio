# Agent Instructions

## Repo Map

This repository is the ab-initio LiF polaron engine (`rmc-ab-initio`) plus its research artifacts. It builds on the IronMC framework, a public repository checked out beside this one as `../ironmc`. `Cargo.toml` reaches `rmc-core`, `rmc-diagmc`, and `rmc-frohlich` there through path dependencies, and `build.rs` embeds both repositories' commits in the binary (`<this>[-dirty]+ironmc-<framework>[-dirty]`).

- `src/`, `tests/`, `examples/`, `build.rs` - the `rmc-ab-initio` crate (CPU slotmap, batched flat, and CubeCL GPU engines; vertex tables; EPW readers).
- `configs/` - run configs (`lif-<engine>[-tauN].json`), see `configs/README.md`.
- `datasets/LiF/` - QE/EPW inputs and outputs, and the vertex tables (`*.rmcvtab`, gitignored).
- `nu/ironmc` - nushell CLI: run targets (`cpu`/`gpu run`), tables, dispersion sweeps, and VSC-5 slurm dispatch (`slurm submit`).
- `runs/` (gitignored) - experiment dirs from `slurm submit --out-dir runs/<experiment>` (required on every submission): `jobs/<stamp>-<name>/` (scripts + logs, one per submission) beside `results/` (the run's cumulative dumps; resubmitting the same --out-dir resumes the checkpointed run). Run targets read the run dir from `$IRONMC_OUT_DIR`, which submit exports. Locally, `runs/` is a pull-only mirror of the cluster (`make sync`).
- `notebooks/` - analysis runbooks (E₀ extrapolation, dispersion/mass).
- `latex/` - methodology note, glossary, symmetry-paths note.
- `journal.md` - implementation and results log.
- `vsc5/` - cluster installation.

## Framework changes

Changes to `rmc-core`/`rmc-diagmc`/`rmc-frohlich` belong in the `../ironmc` repository. Commit them there, keeping ab-initio specifics (LiF, datasets, cluster workflow) out of that repo's code and docs, since it is public. A binary built against a dirty framework checkout records `-dirty` for it.

## Build & Test

- Never run the full test suite yourself. When full-suite verification is needed, give the user the exact `rtk cargo test --release` command and wait for their results.
- The debug profile is off limits by default. Run Cargo builds, checks, tests, and applications with `--release`. Use debug only after a concrete problem requires it, and state that reason first.
- Run only the smallest relevant release-mode test/check target yourself.
- gpu-cuda builds use their own target dir (`target-cuda/`, `CUDA_TARGET_DIR` in `nu/ironmc/common.nu`); never build the gpu-cuda feature set into `target/`.

## VSC-5 devel QoS — do not underestimate it

Both devel QoS are full production nodes, only walltime-capped (10 min). Measured: one zen3_0512_devel node builds the n=65 rmcvtab table in 5m21s.

- `zen3_0512_devel`: 5 nodes, each 128 cores + 512 GB RAM.
- `zen3_0512_a100x2_devel`: 2 nodes, each 512 GB RAM + 2× A100 (4 GPUs total).

Sizing that fits comfortably on one 512 GB node is devel-viable if it finishes (or checkpoints) within 10 minutes — don't route work to the normal QoS queue just because it "sounds big". On the cluster this repo lives at `$DATA/ironmc-ab-initio`, next to the framework at `$DATA/ironmc`.

## Shell

Prefix shell commands with `rtk`.

Examples:

```bash
rtk git status
rtk cargo test --release --test datasets_lif
```

## Benchmarks and cluster jobs

Do not run benchmarks or submit cluster jobs yourself. Benchmark commands may require `sudo` to isolate a CPU core. When benchmark or cluster verification is needed, tell the user exactly which command to run and wait for their results.

## Conventions

- Plan files live in `plans/{open,active,done}/*.md` (gitignored, local), sorted by status: `open` = not started, `active` = partly done, `done` = implemented.
- Prefer existing `type: summary` commit subjects such as `feat:`, `fix:`, `chore:`, `refactor:`, and `docs:`.
- Document implementation chunks in journal.md and physics notes and methodology in ./latex/ab-initio-diagmc/ab-initio-diagmc.tex
- Project terminology is defined in ./latex/glossary/glossary.tex — keep it current when new terms appear.
- Do not put plan checkpoint names (A1, B3, etc.) anywhere else. Plan files are ephemeral and may not exist at a later point.
