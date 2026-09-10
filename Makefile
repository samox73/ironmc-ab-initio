# Rust Froehlich-polaron DiagMC (rmc-frohlich)
#
#   make run                 run the release build against crates/apps/rmc-frohlich/input.json
#
# Performance targets (all use release-tuned + target-cpu=native, pinned to core 0):
#   make test-perf-frohlich  end-to-end polaron sim in runs/test-perf-froehlich/alpha-5/
#   make test-perf-minimal   framework (bare) + physics (full) step-rate measurement
#   make perf                run both perf tests in sequence
#
# Criterion micro-benchmarks (rmc-core hot path):
#   make bench               run benchmarks, no baseline
#   make bench-before        run benchmarks and save results as the "before" baseline
#   make bench-after         run benchmarks and compare against the "before" baseline

SHELL := bash

CRATE              := rmc-frohlich
PERF_DIR           := runs/test-perf-froehlich/alpha-5
TUNED_BIN          := $(CURDIR)/target/release-tuned/$(CRATE)

MINIMAL_CRATE      := rmc-minimal
MINIMAL_TUNED_BIN  := $(CURDIR)/target/release-tuned/$(MINIMAL_CRATE)
MINIMAL_STEPS      ?= 100000000
MINIMAL_WARMUP     ?= 1000000

VSC_HOST           := samuel_soap@vsc5.vsc.ac.at
VSC_PORT           := 27
VSC_SSHFS_OPTIONS  := ServerAliveInterval=15,ServerAliveCountMax=3

# Flags shared by all tuned builds and benchmarks.
TUNED_FLAGS        := RUSTFLAGS="-C target-cpu=native"

.DEFAULT_GOAL := run
.PHONY: run sync sync-vsc sync-replica mount-vsc unmount-vsc bench bench-core bench-minimal bench-frohlich

# Default: run the release build using the crate's committed input.json.
# Results are written to ./results (the binary's default output directory).
run:
	cargo run --release -p $(CRATE) -- crates/apps/$(CRATE)/input.json

sync: sync-replica sync-vsc

sync-vsc:
	@data="$$(ssh -S none -p $(VSC_PORT) $(VSC_HOST) printenv DATA)" && test -n "$$data" && rsync -avP -e 'ssh -S none -p $(VSC_PORT)' --exclude=/.git/ --exclude=/target/ --exclude=/profile/ --exclude=/results/ --exclude=/results-rung1/ --exclude=/slurm/ ./ "$(VSC_HOST):$$data/ironmc/"

sync-replica:
	rsync -avP --exclude=/target/ --exclude=/results/ --exclude=/slurm/ ./ recker@replica.mpi.univie.ac.at:~/ironmc/

mount-vsc:
	@data="$$(ssh -p $(VSC_PORT) $(VSC_HOST) printenv DATA)" && \
		test -n "$$data" && \
		mkdir -p "$(CURDIR)/slurm" "$(CURDIR)/results" && \
		ssh -p $(VSC_PORT) $(VSC_HOST) mkdir -p "$$data/ironmc/slurm" "$$data/ironmc/results" && \
		{ mountpoint -q "$(CURDIR)/slurm" || sshfs -p $(VSC_PORT) -o $(VSC_SSHFS_OPTIONS) "$(VSC_HOST):$$data/ironmc/slurm" "$(CURDIR)/slurm"; } && \
		{ mountpoint -q "$(CURDIR)/results" || sshfs -p $(VSC_PORT) -o $(VSC_SSHFS_OPTIONS) "$(VSC_HOST):$$data/ironmc/results" "$(CURDIR)/results"; }

unmount-vsc:
	@if mountpoint -q "$(CURDIR)/slurm"; then fusermount3 -u "$(CURDIR)/slurm"; fi
	@if mountpoint -q "$(CURDIR)/results"; then fusermount3 -u "$(CURDIR)/results"; fi

bench: bench-core bench-minimal bench-frohlich

bench-core:
	@cargo bench-compare --bench hot_path --dedicate-core
	@echo "------------------------------------------------------"

bench-minimal:
	@cargo bench-compare --bin rmc-minimal --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --progress-regex 'step (\d+)/(\d+)' --dedicate-core -- bare 3000000
	@echo "------------------------------------------------------"
	@cargo bench-compare --bin rmc-minimal --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --progress-regex 'step (\d+)/(\d+)' --dedicate-core -- full 3000000
	@echo "------------------------------------------------------"

bench-frohlich:
	@cargo bench-compare --bin rmc-frohlich --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --dedicate-core -- bench
	@echo "------------------------------------------------------"

build-vsc:
	@RUSTFLAGS="-C target-cpu=znver3" cargo build --release -p rmc-ab-initio
