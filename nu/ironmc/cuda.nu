# Userspace CUDA toolkit for the cubecl-cuda backend (toolkit >= 12 required).
#
# Cluster modules often stop at 11.8 while the node driver already supports 12.x;
# the toolkit is just files, so install it into user storage (no root — the driver
# part of the runfile is skipped) and point CUDA_PATH/LD_LIBRARY_PATH at it:
#   ironmc cuda setup        # once per machine
#   ironmc cuda load         # once per shell, before cargo ... --features gpu-cuda
#
# Install root: $IRONMC_CUDA_ROOT if set, else $DATA (VSC clusters: HOME quotas are
# far too small for the ~7 GB toolkit + 4.2 GB runfile; $DATA is GPFS and visible
# on every node), else $HOME.

# Known runfile URLs; other versions via `setup --url` (archive page:
# https://developer.nvidia.com/cuda-toolkit-archive, Linux > x86_64 > runfile local).
const RUNFILES = {
    "12.6": "https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run"
    "12.4": "https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run"
}

def install-root [] {
    if "IRONMC_CUDA_ROOT" in $env {
        $env.IRONMC_CUDA_ROOT
    } else if "DATA" in $env {
        $env.DATA
    } else {
        $env.HOME
    }
}

def default-prefix [version: string] {
    install-root | path join opt $"cuda-($version)"
}

# Highest CUDA version the node's driver supports, or null without nvidia-smi.
def driver-cap [] {
    if (which nvidia-smi | is-empty) { return null }
    let caps = (nvidia-smi | parse -r 'CUDA Version:\s*(?<v>[0-9]+\.[0-9]+)')
    if ($caps | is-empty) { null } else { $caps | first | get v }
}

# Download the CUDA runfile and install the toolkit (only) into the install root.
export def setup [
    version: string = "12.6"  # toolkit line; known URLs: 12.6, 12.4
    --url: string             # explicit runfile URL for other versions
    --root: string            # explicit install root (overrides $IRONMC_CUDA_ROOT/$DATA/$HOME)
] {
    let dest = if $root != null {
        $root | path join opt $"cuda-($version)"
    } else {
        default-prefix $version
    }
    if ($dest | path join bin nvcc | path exists) {
        print $"already installed at ($dest)"
        return $dest
    }

    let cap = (driver-cap)
    if $cap != null and ($cap | into float) < ($version | into float) {
        error make {msg: $"driver supports CUDA ($cap) at most; toolkit ($version) could not run here"}
    }

    let url = if $url != null {
        $url
    } else if $version in $RUNFILES {
        $RUNFILES | get $version
    } else {
        error make {msg: $"no known runfile for ($version); pass --url from the CUDA archive page"}
    }

    let tmp = ($dest | path dirname | path dirname | path join tmp)
    mkdir $tmp
    let runfile = ($tmp | path join ($url | path basename))
    curl -L -C - -o $runfile $url
    sh $runfile --silent --toolkit $"--toolkitpath=($dest)" $"--tmpdir=($tmp)"
    ^($dest | path join bin nvcc) --version
    print $"installed to ($dest); ($runfile) kept for reinstalls, delete to reclaim ~4.5 GB"
    $dest
}

# Export CUDA_PATH/PATH/LD_LIBRARY_PATH for an installed toolkit (idempotent).
# cudarc detects the toolkit via CUDA_PATH at build time; libnvrtc is dlopened at run time.
export def --env load [
    version: string = "12.6"
    --path: string  # explicit toolkit prefix, overrides the default install-root location
] {
    let dest = if $path != null { $path } else { default-prefix $version }
    if not ($dest | path join include cuda.h | path exists) {
        error make {msg: $"no CUDA toolkit at ($dest) — run `ironmc cuda setup` first"}
    }

    $env.CUDA_PATH = $dest
    $env.PATH = ($env.PATH | prepend ($dest | path join bin) | uniq)
    let lib = ($dest | path join lib64)
    let old = if "LD_LIBRARY_PATH" in $env {
        $env.LD_LIBRARY_PATH | split row ":"
    } else {
        []
    }
    $env.LD_LIBRARY_PATH = ($old | prepend $lib | where {|p| $p != "" } | uniq | str join ":")
    print $"CUDA ($version) at ($dest) \(driver cap: (driver-cap | default 'unknown')\)"
}
