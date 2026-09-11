#!/usr/bin/env nu

let bin_dir = ($env.DATA | path join "bin")
mkdir $bin_dir

# Toolchain goes to the default ~/.cargo and ~/.rustup — the install that
# `ironmc slurm submit` (which does the compiling) and the jobs resolve.
# Bootstrap only when none exists yet — rustup-init refuses to run over a
# live installation.
if not ($env.HOME | path join ".cargo/bin/rustup" | path exists) {
    let rustup_init = ($bin_dir | path join "rustup-init")
    if not ($rustup_init | path exists) {
        http get https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init
            | save --force $rustup_init
        ^chmod +x $rustup_init
    }
    ^$rustup_init -y --no-modify-path --profile minimal --default-toolchain stable
}
