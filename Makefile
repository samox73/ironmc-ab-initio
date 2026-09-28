# ironmc-ab-initio: cluster sync, mounts, and builds.
#
#   make sync         push code to replica + VSC-5, pull VSC-5 runs/ back
#   make mount-vsc    sshfs-mount VSC-5's runs/ at ./runs-vsc (unmount-vsc to undo)
#   make build-vsc    release build tuned for the VSC-5 zen3 nodes
#
# The framework is the sibling checkout ../ironmc (Cargo.toml path deps). The
# cluster mirrors that layout: $DATA/ironmc-ab-initio (this repo, datasets,
# runs) next to $DATA/ironmc (framework sources only), so every sync pushes both.

# Recipes are nushell. Make expands $(VAR) before nu sees the line, so any $
# meant for nushell (variables, $"...") must be written $$ to survive make.
# .ONESHELL runs each recipe as one nu script, so `let` bindings persist across
# lines; --no-config-file skips loading the user's env.nu/config.nu.
SHELL := nu
.SHELLFLAGS := --no-config-file -c
.ONESHELL:

VSC_HOST           := samuel_soap@vsc5.vsc.ac.at
VSC_PORT           := 27
VSC_SSHFS_OPTIONS  := ServerAliveInterval=15,ServerAliveCountMax=3
REPLICA            := recker@replica.mpi.univie.ac.at

.DEFAULT_GOAL := sync

.PHONY: sync sync-vsc sync-replica mount-vsc unmount-vsc build-vsc

sync: sync-replica sync-vsc

# The cargo-relevant sources (this repo's and the framework's) are pushed with
# --no-times --checksum: preserved local mtimes (often older than the cluster's
# cargo fingerprints) make cargo skip rebuilding freshly synced source. Changed
# files land with mtime = now, so cargo rebuilds exactly what changed;
# --checksum keeps unchanged files untransferred (their old mtimes stay put).
# Only the cargo sources get this — checksumming the whole tree would read
# datasets/ (~17 GB) on both ends. The framework's .git goes along so build.rs
# on the cluster records the framework commit the sources came from.
sync-vsc:
	@let data = (ssh -S none -p $(VSC_PORT) $(VSC_HOST) printenv DATA | str trim)
	if ($$data | is-empty) { error make { msg: "DATA is empty on $(VSC_HOST) — is it set?" } }
	print "\nUploading the framework to VSC5"
	rsync -avP --no-times --checksum -e 'ssh -S none -p $(VSC_PORT)' --relative ../ironmc/./crates ../ironmc/./Cargo.toml ../ironmc/./Cargo.lock ../ironmc/./.git $$"$(VSC_HOST):($$data)/ironmc/"
	print "\nUploading ironmc-ab-initio to VSC5"
	rsync -avP --no-times --checksum -e 'ssh -S none -p $(VSC_PORT)' --relative ./src ./tests ./examples ./build.rs ./Cargo.toml ./Cargo.lock $$"$(VSC_HOST):($$data)/ironmc-ab-initio/"
	let excludes = ["--exclude=/target/" "--exclude=/target-cuda/" "--exclude=/.worktrees/" "--exclude=/runs/" "--exclude=/runs-vsc/" "--exclude=/src/" "--exclude=/tests/" "--exclude=/examples/" "--exclude=/build.rs" "--exclude=/Cargo.toml" "--exclude=/Cargo.lock" "--exclude=*.rmcvtab" "--exclude=*.rmcvtab.shard-*"]
	rsync -avP -e 'ssh -S none -p $(VSC_PORT)' ...$$excludes ./ $$"$(VSC_HOST):($$data)/ironmc-ab-initio/"
	print "\nDownloading from VSC5"
	rsync -avP -e 'ssh -S none -p $(VSC_PORT)' "--exclude=*.rmcvtab" "--exclude=*.rmcvtab.shard-*" "--exclude=batches-rank*.bin" "--exclude=checkpoint-rank*.bin" $$"$(VSC_HOST):($$data)/ironmc-ab-initio/runs" ./

sync-replica:
	rsync -avP --exclude=/target/ --exclude=/mcforge/ ../ironmc/ $(REPLICA):~/ironmc/
	rsync -avP --exclude=/target/ --exclude=/target-cuda/ --exclude=/.worktrees/ --exclude=/runs/ --exclude=/runs-vsc/ ./ $(REPLICA):~/ironmc-ab-initio/

# runs/ mounts at ./runs-vsc, not ./runs: sync-vsc rsyncs into ./runs, and a
# pull into a live sshfs mount would write straight back to the cluster.
mount-vsc:
	@def mounted [p] { (mountpoint -q $$p | complete).exit_code == 0 }
	let data = (ssh -p $(VSC_PORT) $(VSC_HOST) printenv DATA | str trim)
	if ($$data | is-empty) { error make { msg: "DATA is empty on $(VSC_HOST) — is it set?" } }
	mkdir "$(CURDIR)/runs-vsc"
	ssh -p $(VSC_PORT) $(VSC_HOST) mkdir -p $$"($$data)/ironmc-ab-initio/runs"
	if not (mounted "$(CURDIR)/runs-vsc") { sshfs -p $(VSC_PORT) -o $(VSC_SSHFS_OPTIONS) $$"$(VSC_HOST):($$data)/ironmc-ab-initio/runs" "$(CURDIR)/runs-vsc" }

unmount-vsc:
	@def mounted [p] { (mountpoint -q $$p | complete).exit_code == 0 }
	if (mounted "$(CURDIR)/runs-vsc") { fusermount3 -u "$(CURDIR)/runs-vsc" }

build-vsc:
	@RUSTFLAGS="-C target-cpu=znver3" cargo build --release
