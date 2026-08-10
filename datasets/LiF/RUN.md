# LiF QE + EPW run playbook

Environment: **QE 7.5 / EPW 6.0** (NixOS profile, auto-updated — check `pw.x` version
before a campaign!). Gather script: **`epw_pp.py`**.

> 🛑 **Use NORM-CONSERVING pseudopotentials.** The current deck uses ultrasoft
> (SSSP-eff GBRV) → EPW's e-ph tensor comes out all-NaN (finite only at q=Γ). EPW
> only supports NC pseudos for e-ph. Full diagnosis: **[`EPH-NAN-DIAGNOSIS.md`](EPH-NAN-DIAGNOSIS.md)**.

> ⚠️ **ONE QE version for the whole chain.** scf/ph/nscf/epw must all be the same QE
> release. The system upgraded 7.4.1→7.5 mid-run; EPW 6.0 then read a 7.4.1
> `_ph0/*.phsave/patterns.xml` whose format had changed → heap corruption / `malloc`
> assertion deep in `ep_coarse_unfolding` (`gmap_sym`/`read_disp_pattern`/`xml_open_file`
> — the frame varies, it's whatever `malloc` runs next). Fix = rerun scf→ph→pp→nscf→epw
> under the current version; `rm -rf save out` first so no stale-version files mix.

`bands.x` available for the Wannier check. **`nscf.in` k-points MUST be an explicit
Γ-centered crystal grid** (`{i/6}`, 216 pts) — EPW's `ktokpmq` folds `k+q` assuming
Γ-centered fractional coords. `K_POINTS automatic 6 6 6 0 0 0` gives a *half-shifted*
even MP mesh → `coarse k-mesh needs to be strictly positive in 1st BZ` abort. `nosym`
alone does NOT fix this. Equivalent to `kmesh.pl 6 6 6` output.

All commands run from this directory:

```bash
cd /home/samox/studies/phd/repositories/ironmc/datasets/LiF
```

Parallel: prefix `mpirun -np N`; add `-npool K` to `pw.x`/`ph.x`.
**For EPW, `-npool` must equal the number of MPI ranks.** Keep pools consistent across ph/nscf/epw.

---

## 0 · Relax the lattice  (DONE — celldm(1) = 7.68461695226161 bohr / 4.067 Ang)

```bash
pw.x -in relax.in > relax.out
grep "JOB DONE" relax.out
grep "new unit-cell volume" relax.out | tail -1      # V_prim, in a.u.^3 (= bohr^3)
grep -A5 "Begin final coordinates" relax.out          # relaxed CELL_PARAMETERS vectors
```

Do NOT read `celldm(1)` from `"lattice parameter (alat)"` — that line is QE's *fixed reference*
unit and never changes. Get the relaxed constant from the volume (rocksalt primitive cell holds
1 formula unit, `V_prim = a^3/4`):

```bash
python3 -c "V=  <V_prim in bohr^3>; print('celldm(1) =', (4*V)**(1/3), 'bohr')"
```

Put that value into `scf.in` and `nscf.in`. (If `"new unit-cell volume"` never appears, the cell
never took a step — the starting guess was already the minimum.)

## 1 · SCF

```bash
pw.x -in scf.in > scf.out
grep -e "convergence has been achieved" -e "JOB DONE" scf.out
grep "highest occupied" scf.out                       # gap sanity
```

## 2 · Phonons + dvscf   (expensive: 6x6x6 -> ~16 irreducible q, each a DFPT)

```bash
ph.x -in ph.in > ph.out
grep "JOB DONE" ph.out
grep "freq (" ph.out          # acoustic at Gamma ~ 0 OK; any large NEGATIVE freq = problem
grep -e "Dielectric constant" -e "Effective charges" ph.out   # epsil / Born charges ran (for lpolar)
```

## 3 · Gather ph.x output for EPW  ->  ./save

`epw_pp.py` assumes `outdir='.'` — it reads `LiF.save/`, `_ph0/`, `LiF.dyn*` from cwd.
Our deck uses `outdir='./out'`, so first expose those in cwd (dyn files already land there).
It also ships with a CRLF shebang, so call `python3` on it directly (env chokes on `python3\r`).

```bash
ln -sfn out/LiF.save LiF.save
ln -sfn out/_ph0 _ph0
echo LiF | python3 $(which epw_pp.py)     # NOT `epw_pp.py` (CRLF shebang) 
ls save/                                  # expect LiF.dyn*, LiF.dvscf*, patterns
```
(`run_pipeline.nu` does all of this for you.)

## 4 · NSCF (full uniform 6x6x6 grid) -> read band edges for the Wannier windows

```bash
pw.x -in nscf.in > nscf.out
grep "JOB DONE" nscf.out
grep "highest occupied, lowest unoccupied" nscf.out   # VBM / CBM in eV
```

Then TUNE `epw.in` from the CBM:
- `dis_froz_min ~ CBM - 0.2`
- `dis_froz_max ~` top of the lowest isolated conduction band (scan nscf eigenvalues, a few eV above CBM)
- `dis_win_max  ~` a few eV above `dis_froz_max`

## 5 · EPW -> the tensors patt126 reads

```bash
epw.x -npool $N -in epw.in > epw.out       # match -np
grep "JOB DONE" epw.out
ls -la epwdata.fmt LiF.epmatwp LiF.epb1     # the three downstream files
```

---

## Two things that need your eyes (not just a green JOB DONE)

1. **Phonons real** (step 2): a spurious imaginary mode means lattice/cutoffs are off — fix before EPW.
2. **Wannier fit** (step 5): EPW writes `LiF_band.dat` (Wannier-interpolated along L-G-X). Overlay it on
   a DFT band structure; if the **CBM region** doesn't match, adjust `proj`/`dis_*` and **rerun only
   step 5** (EPW re-Wannierizes fast; QE stages don't repeat). A bad fit silently corrupts every g(k,q).

## Downstream handoff

When `epwdata.fmt`, `LiF.epmatwp`, `LiF.epb1` all exist, point the patt126 readers/notebooks at
`dataset/LiF/` with `prefix = "LiF"` (they currently hardcode `"silicon"`). That's the A0 -> A1
handoff for `export_compressed(...)` feeding `rmc-ab-initio`.
