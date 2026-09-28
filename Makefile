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

# Recipes are nushell. Make expands $(VAR) before nu sees the line, so any $
# meant for nushell (variables, $"...") must be written $$ to survive make.
# .ONESHELL runs each recipe as one nu script, so `let` bindings persist across
# lines; --no-config-file skips loading the user's env.nu/config.nu.
SHELL := nu
.SHELLFLAGS := --no-config-file -c
.ONESHELL:

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
.PHONY: run sync sync-vsc sync-replica mount-vsc unmount-vsc bench bench-core bench-minimal bench-frohlich build-vsc

# Default: run the release build using the crate's committed input.json.
# Results are written to ./results (the binary's default output directory).
run:
	cargo run --release -p $(CRATE) -- crates/apps/$(CRATE)/input.json

sync: sync-replica sync-vsc

# The cargo-relevant sources are pushed with --no-times --checksum: preserved
# local mtimes (often older than the cluster's cargo fingerprints) make cargo
# skip rebuilding freshly synced source. Changed files land with mtime = now, so
# cargo rebuilds exactly what changed; --checksum keeps unchanged files
# untransferred (their old mtimes stay put). Only crates/ + Cargo.* get this —
# checksumming the whole tree would read datasets/ (~17 GB) on both ends.
sync-vsc:
	@let data = (ssh -S none -p $(VSC_PORT) $(VSC_HOST) printenv DATA | str trim)
	if ($$data | is-empty) { error make { msg: "DATA is empty on $(VSC_HOST) — is it set?" } }
	print "\nUploading to VSC5"
	rsync -avP --no-times --checksum -e 'ssh -S none -p $(VSC_PORT)' --relative ./crates ./Cargo.toml ./Cargo.lock $$"$(VSC_HOST):($$data)/ironmc/"
	let excludes = ["--exclude=/target/" "--exclude=/target-cuda/" "--exclude=/profile/" "--exclude=/runs/" "--exclude=/runs-vsc/" "--exclude=/crates/" "--exclude=/Cargo.toml" "--exclude=/Cargo.lock" "--exclude=*.rmcvtab" "--exclude=*.rmcvtab.shard-*"]
	rsync -avP -e 'ssh -S none -p $(VSC_PORT)' ...$$excludes ./ $$"$(VSC_HOST):($$data)/ironmc/"
	print "\nDownloading from VSC5"
	rsync -avP -e 'ssh -S none -p $(VSC_PORT)' "--exclude=*.rmcvtab" "--exclude=*.rmcvtab.shard-*" "--exclude=batches-rank*.bin" "--exclude=checkpoint-rank*.bin" $$"$(VSC_HOST):($$data)/ironmc/runs" ./

sync-replica:
	rsync -avP --exclude=/target/ --exclude=/target-cuda/ --exclude=/results/ --exclude=/slurm/ --exclude=/runs/ --exclude=/runs-vsc/ ./ recker@replica.mpi.univie.ac.at:~/ironmc/

# runs/ mounts at ./runs-vsc, not ./runs: sync-vsc rsyncs into ./runs, and a
# pull into a live sshfs mount would write straight back to the cluster.
mount-vsc:
	@def mounted [p] { (mountpoint -q $$p | complete).exit_code == 0 }
	let data = (ssh -p $(VSC_PORT) $(VSC_HOST) printenv DATA | str trim)
	if ($$data | is-empty) { error make { msg: "DATA is empty on $(VSC_HOST) — is it set?" } }
	mkdir "$(CURDIR)/slurm" "$(CURDIR)/results" "$(CURDIR)/runs-vsc"
	ssh -p $(VSC_PORT) $(VSC_HOST) mkdir -p $$"($$data)/ironmc/slurm" $$"($$data)/ironmc/results" $$"($$data)/ironmc/runs"
	if not (mounted "$(CURDIR)/slurm") { sshfs -p $(VSC_PORT) -o $(VSC_SSHFS_OPTIONS) $$"$(VSC_HOST):($$data)/ironmc/slurm" "$(CURDIR)/slurm" }
	if not (mounted "$(CURDIR)/results") { sshfs -p $(VSC_PORT) -o $(VSC_SSHFS_OPTIONS) $$"$(VSC_HOST):($$data)/ironmc/results" "$(CURDIR)/results" }
	if not (mounted "$(CURDIR)/runs-vsc") { sshfs -p $(VSC_PORT) -o $(VSC_SSHFS_OPTIONS) $$"$(VSC_HOST):($$data)/ironmc/runs" "$(CURDIR)/runs-vsc" }

unmount-vsc:
	@def mounted [p] { (mountpoint -q $$p | complete).exit_code == 0 }
	for p in ["$(CURDIR)/slurm" "$(CURDIR)/results" "$(CURDIR)/runs-vsc"] { if (mounted $$p) { fusermount3 -u $$p } }

bench: bench-core bench-minimal bench-frohlich

bench-core:
	@cargo bench-compare --bench hot_path --dedicate-core
	print "------------------------------------------------------"

bench-minimal:
	@cargo bench-compare --bin rmc-minimal --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --progress-regex 'step (\d+)/(\d+)' --dedicate-core -- bare 3000000
	print "------------------------------------------------------"
	cargo bench-compare --bin rmc-minimal --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --progress-regex 'step (\d+)/(\d+)' --dedicate-core -- full 3000000
	print "------------------------------------------------------"

bench-frohlich:
	@cargo bench-compare --bin rmc-frohlich --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --dedicate-core -- bench
	print "------------------------------------------------------"

build-vsc:
	@RUSTFLAGS="-C target-cpu=znver3" cargo build --release -p rmc-ab-initio
