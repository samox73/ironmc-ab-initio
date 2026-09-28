#!/usr/bin/env nu
# LiF QE + EPW pipeline for rmc-ab-initio A0.
# Stages:  relax -> scf -> ph -> pp -> nscf -> [manual check] -> epw
#
#   nu run_pipeline.nu                 # scf..epw  (relax skipped; geometry already set)
#   nu run_pipeline.nu --from relax    # full run incl. vc-relax + auto celldm(1) feed
#   nu run_pipeline.nu --from ph       # resume at ph (e.g. after a crash) -- KEEPS save/out
#   nu run_pipeline.nu --np 8          # use 8 MPI ranks/pools (default 10)
#
# Running the whole chain in one go => one QE version everywhere, which is what
# avoids the 7.4.1/7.5 phsave-format mismatch that crashes EPW.  The only manual
# step is a pause after nscf to eyeball / tune the epw.in Wannier windows.

def die  [m: string] { print $"(ansi red)✗ ($m)(ansi reset)"; exit 1 }
def ok   [m: string] { print $"(ansi green)✓ ($m)(ansi reset)" }
def step [m: string] { print $"(ansi cyan)• ($m)(ansi reset)" }

# run a QE binary, capture stdout+stderr to <tag>.out, require "JOB DONE"
def qe-run [tag: string, bin: string, infile: string, np: int] {
    step $"($tag): mpirun -np ($np) ($bin) -in ($infile)"
    mpirun -np $np $bin -npool $np -in $infile out+err> $"($tag).out"
    if (open $"($tag).out" | str contains "JOB DONE") {
        ok $"($tag) JOB DONE"
    } else {
        print (open $"($tag).out" | lines | last 25 | str join (char nl))
        die $"($tag) failed — no JOB DONE (tail above, full log in ($tag).out)"
    }
}

# QE version string from a *.out header, or null if absent
def qe-ver [f: string] {
    if not ($f | path exists) { return null }
    let m = (open $f | lines | where ($it | str contains "Program ")
             | get 0? | default "" | parse -r 'v\.(?<v>[\d.]+)')
    if ($m | is-empty) { null } else { $m.v.0 }
}

# on resume, warn if pre-existing outputs disagree on QE version (the EPW-crash trap)
def check-versions [] {
    let vs = ([scf.out ph.out nscf.out] | each {|f| qe-ver $f} | compact | uniq)
    if ($vs | length) > 1 {
        print $"(ansi yellow)⚠ QE version drift in existing outputs: ($vs)(ansi reset)"
        print "  EPW crashes reading a phsave from a different QE. Recommend a clean run from scf."
        input "  Enter to continue anyway, Ctrl-C to abort... "
    } else if ($vs | length) == 1 {
        ok $"existing outputs are all QE ($vs | get 0)"
    }
}

# vc-relax, then feed the relaxed lattice constant into scf.in + nscf.in.
# Rocksalt primitive cell holds 1 f.u. => V_prim = a^3/4 => a = (4 V)^(1/3).
def do-relax [np: int] {
    qe-run "relax" "pw.x" "relax.in" $np
    let vls = (open relax.out | lines | where ($it | str contains "new unit-cell volume"))
    if ($vls | is-empty) {
        ok "relax: cell already at minimum — celldm(1) unchanged"
        return
    }
    let vol = ($vls | last | parse -r '=\s*(?<v>[\d.]+)' | get v.0 | into float)   # bohr^3
    let a   = ((4 * $vol) ** (1 / 3) | math round --precision 11)
    step $"relaxed volume ($vol) bohr^3  ->  celldm\(1\) = ($a) bohr"
    for f in [scf.in nscf.in] {
        open $f | str replace -r 'celldm\(1\)\s*=\s*[\d.]+' $"celldm\(1\)   = ($a)" | save -f $f
        ok $"updated celldm(1) in ($f)"
    }
}

# gather ph.x output for EPW into ./save via epw_pp.py (answers its prefix prompt).
# Call python3 on the script directly -- epw_pp.py ships with a CRLF shebang that
# makes `env` look for `python3\r`; running python3 <script> bypasses the shebang.
def do-pp [] {
    step "pp: epw_pp.py -> save/"
    # epw_pp.py hardcodes cwd paths (LiF.save/, _ph0/, LiF.dyn*) i.e. assumes outdir='.'.
    # Our deck uses outdir='./out', so save + _ph0 land there while dyn files land in cwd.
    # Expose the two ./out items in cwd via symlinks -- no move, no re-run of ph.
    ^ln -sfn out/LiF.save LiF.save
    ^ln -sfn out/_ph0 _ph0
    let hit = (which epw_pp.py)
    if ($hit | is-empty) { die "epw_pp.py not on PATH" }
    $"LiF(char nl)" | ^python3 ($hit | get path.0)
    let n = (ls save | where name =~ 'dvscf' | length)
    if $n < 1 { die "pp produced no dvscf files in save/ — check epw_pp.py output" }
    ok $"gathered save/ \(($n) dvscf files\)"
}

# the one manual gate: confirm the Wannier windows against the nscf band edges
def pause-before-epw [] {
    print ""
    print "──────── manual check before EPW ────────"
    let edge = (open nscf.out | lines | where ($it | str contains "highest occupied") | last)
    print $"  ($edge | str trim)"
    print "  epw.in Wannier settings:"
    for l in (open epw.in | lines | where ($it =~ 'nbndsub|dis_froz|dis_win|proj\(')) {
        print $"    ($l | str trim)"
    }
    print "  Edit epw.in in another terminal now if the windows need tuning."
    input "  Press Enter to launch EPW... "
}

# EPW needs a big stack (gmap_sym); ulimit is bash-only, so wrap it
def do-epw [np: int, dir: string] {
    step "epw (ulimit -s unlimited)"
    bash -c $"ulimit -s unlimited; cd ($dir); mpirun -np ($np) epw.x -npool ($np) -in epw.in > epw.out 2>&1"
    # EPW prints NO "JOB DONE"; a clean finish ends with "Total program execution".
    # (The nonzero exit code is just the benign IEEE FP signal, not a failure.)
    if not (open epw.out | str contains "Total program execution") {
        print (open epw.out | lines | last 25 | str join (char nl))
        die "epw failed (tail above, full log in epw.out)"
    }
    ok "epw finished"
    # epwdata.fmt lands in cwd; .epmatwp / .epb land in outdir (./out).
    for x in [epwdata.fmt out/LiF.epmatwp out/LiF.epb1] {
        if ($x | path exists) { ok $"wrote ($x)" } else { die $"missing downstream file ($x)" }
    }
    # Finite-check epmatwp: EPW 6.0 can FINISH cleanly yet write LiF.epmatwp as 100%
    # uninitialised heap garbage (NaN) — fail loudly here instead of shipping it. The
    # epb files are already on disk (written above), so the A1 consumer can still fall
    # back to the coarse epb even when this dies. (We do NOT all-finite the epb itself:
    # it legitimately carries NaN in the out-of-window bands we don't use.)
    let chk = (^python3 -c "import numpy as np,sys; a=np.fromfile('out/LiF.epmatwp',dtype=np.float64,count=200000); sys.exit(0 if a.size and np.isfinite(a).all() else 3)" | complete)
    if $chk.exit_code == 0 { ok "epmatwp finite" } else { die "LiF.epmatwp is NaN/garbage — EPW wrote an uninitialised e-ph tensor (run finished but epmatwp unusable; see epw.out). The coarse epb is still on disk for the A1 fallback." }
}

def main [
    --from: string = "scf"   # relax | scf | ph | pp | nscf | epw
    --np: int = 10           # MPI ranks (= pools; EPW requires them equal)
    --dir: string = "/home/samox/studies/phd/repositories/ironmc-ab-initio/datasets/LiF"
] {
    cd $dir
    let stages = [relax scf ph pp nscf epw]
    if $from not-in $stages { die $"--from must be one of: ($stages | str join ', ')" }
    let start = ($stages | enumerate | where item == $from | first | get index)

    print $"(ansi cyan)LiF pipeline — ($stages | skip $start | str join ' -> ')  @ ($np) ranks(ansi reset)"

    if $from in [relax scf] {
        step "cleaning stale save/ out/ + pp symlinks (prevents cross-version phsave mixing)"
        rm -rf save out LiF.save _ph0
    } else {
        check-versions
    }

    for s in ($stages | skip $start) {
        match $s {
            "relax" => { do-relax $np }
            "scf"   => { qe-run "scf" "pw.x" "scf.in" $np }
            "ph"    => { qe-run "ph"  "ph.x" "ph.in"  $np }
            "pp"    => { do-pp }
            "nscf"  => { qe-run "nscf" "pw.x" "nscf.in" $np; pause-before-epw }
            "epw"   => { do-epw $np $dir }
        }
    }
    ok "pipeline complete — epwdata.fmt / LiF.epmatwp / LiF.epb1 ready for export (A1)"
}
