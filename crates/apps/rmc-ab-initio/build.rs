use std::process::Command;

/// Embed the git provenance of the code actually compiled into the binary —
/// submit-time HEAD can be newer than a prebuilt binary, so build time is the
/// only honest recording point.
fn main() {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let hash = git(&["rev-parse", "--short=9", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let dirty = git(&["status", "--porcelain"]).map(|s| !s.is_empty()).unwrap_or(false);
    let code = if dirty { format!("{hash}-dirty") } else { hash };
    println!("cargo:rustc-env=IRONMC_GIT_HASH={code}");

    // Recompile the provenance only when HEAD actually moves (a commit/checkout),
    // watching HEAD's RESOLVED path. Two hazards this avoids, both of which force
    // an in-job recompile that — under multi-rank `srun` — makes N cargo
    // processes race on the shared target/ and corrupt the .rlib:
    //   * `.git/index` (the old trigger) is rewritten by `git status` above,
    //     especially on a dirty tree, so watching it makes build.rs invalidate
    //     itself on essentially every build. Never watch it.
    //   * In a detached `git worktree` (the `--commit-hash` path) `.git` is a
    //     *file*, so the old relative `../../../.git/HEAD` did not exist — cargo
    //     treats a missing watched path as always-changed and rebuilds every
    //     time. `--absolute-git-dir` resolves the worktree's real gitdir.
    // HEAD itself is not rewritten by `git status`/staging, so its mtime is
    // stable between a login-node prebuild and the compute-node job.
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
    }
}
