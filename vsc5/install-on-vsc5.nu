#!/usr/bin/env nu

let host = "samuel_soap@vsc5.vsc.ac.at"
let port = 27
let temp = (^mktemp -d | str trim)
let control = ($temp | path join "ssh")
print "Authenticating with VSC5"
^ssh -p $port -M -S $control -o ControlPersist=10m -fN $host

let remote_data = (^ssh -p $port -S $control $host printenv DATA | str trim)
let remote_home = (^ssh -p $port -S $control $host printenv HOME | str trim)
if ($remote_data | is-empty) or ($remote_home | is-empty) {
    error make {msg: "VSC5 did not report DATA or HOME"}
}

let repo = ($env.FILE_PWD | path dirname)
let remote_repo = $"($remote_data)/ironmc"
let bin_dir = $"($remote_home)/.local/bin"
^ssh -p $port -S $control $host mkdir -p $remote_repo $bin_dir

let excludes = ["--exclude=.git" "--exclude=target" "--exclude=profile" "--exclude=results" "--exclude=results-stage1" "--exclude=slurm"]
print "Transferring files with rsync"
^rsync --archive --info=progress2 --rsh $"ssh -p ($port) -S ($control)" ...$excludes $"($repo)/" $"($host):($remote_repo)/"

# Static musl builds pushed to ~/.local/bin, sha256-pinned.
let tools = [
    [bin, url, sha256];
    [nu, "https://github.com/nushell/nushell/releases/download/0.114.1/nu-0.114.1-x86_64-unknown-linux-musl.tar.gz", "ce3a1c5a07c784098b5675224a165e73a45f0f720a8392791d7c3a5b4720255e"]
    [fd, "https://github.com/sharkdp/fd/releases/download/v10.5.0/fd-v10.5.0-x86_64-unknown-linux-musl.tar.gz", "761c72dc8e120d85b22292063be8a796e2eeb20eb3e4f38b8fa2343ccf3514a7"]
    [rg, "https://github.com/BurntSushi/ripgrep/releases/download/15.2.0/ripgrep-15.2.0-x86_64-unknown-linux-musl.tar.gz", "33e15bcf1624b25cdd2a55813a47a2f95dbe126268203e76aa6a585d1e7b149c"]
]
for tool in $tools {
    let archive = $"($temp)/($tool.bin).tar.gz"
    http get $tool.url | save --force $archive
    let actual = (open --raw $archive | hash sha256)
    if $actual != $tool.sha256 {
        error make {msg: $"($tool.bin) checksum mismatch: ($actual)"}
    }
    let extract = $"($temp)/($tool.bin)-extract"
    mkdir $extract
    ^tar -xzf $archive -C $extract
    let bin_path = (glob $"($extract)/**/($tool.bin)" | first)
    ^rsync --archive --info=progress2 --rsh $"ssh -p ($port) -S ($control)" $bin_path $"($host):($bin_dir)/($tool.bin)"
    ^ssh -p $port -S $control $host chmod +x $"($bin_dir)/($tool.bin)"
}
^ssh -p $port -S $control $host $"($bin_dir)/nu" $"($remote_repo)/vsc5/install.nu"
^ssh -p $port -S $control -O exit $host

print $"Installed ironmc in ($remote_repo)"
print "Dispatch jobs on VSC5 with: ironmc slurm submit <target> (load module with: use $env.DATA/ironmc/nu/ironmc)"
