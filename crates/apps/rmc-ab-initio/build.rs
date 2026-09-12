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
    println!("cargo:rerun-if-changed=../../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../../.git/index");
}
