# ironmc-ab-initio

First-principles diagrammatic Monte Carlo for the LiF electron polaron, built on the [IronMC](https://github.com/samox73/ironmc) framework. The engine replaces the analytic Fröhlich vertex with electron–phonon matrix elements from DFPT + Wannier interpolation (Quantum ESPRESSO + EPW), compresses them by SVD into vertex tables on an n³ grid, and samples the zero-temperature imaginary-time Green's function with CPU and CUDA engines. The reproduction target is Luo, Park, Bernardi, Nat. Phys. 21, 1275 (2025).

Methodology, validation, and error budgets are in [`latex/ab-initio-diagmc/`](latex/ab-initio-diagmc/ab-initio-diagmc.tex); the implementation and results log is [`journal.md`](journal.md).

## Layout and the framework dependency

The framework must be checked out beside this repository:

```
repositories/
├── ironmc/             # framework (rmc-core, rmc-diagmc, rmc-frohlich, …)
└── ironmc-ab-initio/   # this repo; Cargo.toml path-depends on ../ironmc
```

`build.rs` embeds both commits in the binary, e.g. `655222999+ironmc-6f743f75f` (with `-dirty` for uncommitted changes), and every dump, checkpoint, and result JSON records it.

| Path | Contents |
|---|---|
| `src/`, `tests/`, `examples/` | the `rmc-ab-initio` crate |
| `configs/` | run configs, see [`configs/README.md`](configs/README.md) |
| `datasets/LiF/` | QE/EPW inputs, outputs, and vertex tables (large files gitignored), see [`datasets/LiF/README.md`](datasets/LiF/README.md) |
| `nu/ironmc/` | nushell CLI: runs, tables, dispersion sweeps, VSC-5 slurm dispatch |
| `notebooks/` | analysis runbooks |
| `latex/` | methodology note, glossary, symmetry-paths note |
| `vsc5/` | cluster installation, see [`vsc5/README.md`](vsc5/README.md) |
| `runs/` | experiment dirs (gitignored; mirrored from the cluster) |

## Build and run

```nu
cargo build --release
cargo test --release --test datasets_lif        # dataset sanity
use nu/ironmc; ironmc configs list; ironmc results   # configs side by side; finished runs
```

GPU builds use `--features gpu-cuda` and their own target dir; the `ironmc gpu …` targets handle both.

## Cluster workflow

Code reaches VSC-5 with `make sync` (pushes this repo and the framework sources, pulls `runs/` back). Jobs go through `ironmc slurm submit { … } --out-dir runs/<experiment>`; see `vsc5/README.md`. On the cluster the two repositories sit at `$DATA/ironmc-ab-initio` and `$DATA/ironmc`.
