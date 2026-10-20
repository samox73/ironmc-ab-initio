use std::path::{Path, PathBuf};
use std::process::Command;

/// Embed the git provenance of the code actually compiled into the binary —
/// submit-time HEAD can be newer than a prebuilt binary, so build time is the
/// only honest recording point.
fn main() {
    let git_in = |dir: &Path, args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let git = |args: &[&str]| git_in(&here, args);
    let hash = git(&["rev-parse", "--short=9", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let dirty = git(&["status", "--porcelain"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let own = if dirty { format!("{hash}-dirty") } else { hash };

    // The framework is a path dependency on the sibling ../ironmc checkout
    // (Cargo.toml), so its commit is part of the compiled code too. Dirty
    // means tracked changes under what the build reads (crates/ + Cargo.*);
    // untracked scratch in that repo does not count.
    let framework = here.join("../ironmc");
    let fw_hash = git_in(&framework, &["rev-parse", "--short=9", "HEAD"])
        .unwrap_or_else(|| "unknown".to_string());
    let fw_dirty = git_in(
        &framework,
        &[
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
        ],
    )
    .map(|s| !s.is_empty())
    .unwrap_or(false);
    let fw = if fw_dirty { format!("{fw_hash}-dirty") } else { fw_hash };
    println!("cargo:rustc-env=IRONMC_GIT_HASH={own}+ironmc-{fw}");

    // Recompile the provenance only when HEAD actually moves, watching the HEAD
    // *reflog*: appended on every commit/checkout/reset, untouched by `git
    // status`/staging, so its mtime is stable between a login-node prebuild and
    // the compute-node job. Hazards this dodges, each of which forces an in-job
    // recompile that — under multi-rank `srun` — makes N cargo processes race on
    // the shared target/ and corrupt the .rlib:
    //   * `.git/index` is rewritten by `git status` above, especially on a dirty
    //     tree, so watching it invalidates build.rs on essentially every build.
    //   * `.git/HEAD` alone misses same-branch commits (they move the branch
    //     ref, not HEAD — its mtime provably doesn't change), freezing the hash
    //     at the last checkout. Only the fallback when the reflog is disabled.
    //   * In a detached `git worktree` (the `--commit-hash` path) `.git` is a
    //     *file*; cargo treats a missing watched path as always-changed.
    //     `--absolute-git-dir` resolves the worktree's real gitdir, which has
    //     its own logs/HEAD.
    // Same rule for the framework checkout: rerun when its HEAD moves.
    for dir in [&here, &framework] {
        if let Some(git_dir) = git_in(dir, &["rev-parse", "--absolute-git-dir"]) {
            let reflog = format!("{git_dir}/logs/HEAD");
            if std::path::Path::new(&reflog).exists() {
                println!("cargo:rerun-if-changed={reflog}");
            } else {
                println!("cargo:rerun-if-changed={git_dir}/HEAD");
            }
        }
    }
}
