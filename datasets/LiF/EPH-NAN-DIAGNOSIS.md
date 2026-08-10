# Diagnosis: EPW e-ph tensor is NaN (A0 LiF)

**Status: root cause found — 2026-08-10.**

> **TL;DR.** `LiF.epmatwp` (and the coarse `LiF.epb*`) contain no usable
> electron-phonon data: the coupling is finite **only at q=Γ** and NaN at every
> other q. The cause is **our pseudopotential choice** — LiF was built with
> **ultrasoft** (SSSP-eff GBRV) pseudos, but **EPW's e-ph matrix elements are only
> supported with norm-conserving pseudopotentials.** Fix = swap Li+F to
> norm-conserving pseudos and redo the whole chain (scf→ph→nscf→epw). Phonons,
> `dvscf`, wavefunctions and the dynamical matrix are all fine; only EPW's e-ph
> breaks.

This was **not** an EPW bug and **not** the band-window / `gmap_sym` issues we
chased first (see *Dead ends* below). `A0 DONE` in earlier notes was premature —
nobody had checked the tensor *values*; the file was the right size but all-NaN
from the start.

---

## Symptom

| Artifact | State |
|---|---|
| `out/LiF.epmatwp` | 100% NaN. Uniform uninitialised-memory sentinel (`0xfffffffc00000000`, then `0xfffffffffffffffd` on a rerun — two *different* uniform values ⇒ uninitialised heap, not a computed NaN or a `-finit` fill). |
| `out/LiF.epb1..10` (coarse Bloch e-ph) | e-ph finite **only at q-index 0 (Γ)**; all 215 other q are NaN. (`epb` reports "91% finite" only because 200/225 band-pairs are zero-padding; the real 5×5 coupling block is 99.5% NaN.) |
| `decay.epmate`, `decay.epmatp` | `NaN` for every R-vector (EPW's own decay diagnostic). |
| Exit | Clean finish ("Total program execution"); the "IEEE_INVALID" note is this NaN, not benign. |

The finite-only-at-Γ pattern is the key clue: Γ is the trivial reference q (no
star rotation, no k+q remap, no augmentation structure factor), so it survives
whatever breaks every other q.

---

## What we verified is FINE (ruled out)

- **`dvscf_q1..16`** — all present, finite, non-zero, and **distinct** (different
  md5s / magnitudes). The `epw_pp.py` gathering (with the `outdir=./out` symlink
  hack) worked.
- **Phonons / dynamical matrix** — `dynq` in the `epb` reconstructs LiF's phonons
  correctly (acoustic sum rule: 3 zero modes at Γ; TO ≈ 11.9 THz; LO-TO splitting
  present at finite q). `ph.x` fully supports ultrasoft, so the phonon side is
  clean.
- **Electron Hamiltonian** — `epwdata.fmt` `H(R)` gives CBM = 9.463 eV at Γ with
  the correct Li-s / Li-p₃ structure. Wavefunctions/Wannierisation are fine.
- **k-grid** — nscf is the standard 216-point Γ-centred `i/6` crystal list
  (`[0, 5/6]`, nested-loop order); EPW sees all 216 k in 10 pools.
- **Cutoffs / FFT grids** — `ecutwfc=50`, `ecutrho=400`, dense `36³` / smooth
  `25³` grids are **identical** across scf, ph, nscf, epw. No stage mismatch.
- **QE versions** — scf/ph/nscf all v7.5; EPW 6.0 is the matched pair. Not the
  phsave version-mismatch trap.

---

## Root cause: ultrasoft pseudopotentials

`scf.out`: *"Pseudo is Ultrasoft, Zval = 3.0"* (Li) / *"= 7.0"* (F) — the SSSP-eff
GBRV set. **EPW's electron-phonon matrix elements require norm-conserving
pseudopotentials.** Ultrasoft (and PAW) are unsupported for the e-ph: the
augmentation contribution `⟨ψ_{k+q}| ΔV_aug |ψ_k⟩` is not handled, and ultrasoft
is what forces `ecutrho=400` → the dense `36³` grid the G-mapping fragility rode
on.

Why it matches every observation:

- `ph.x` supports ultrasoft → correct `dvscf`, dynmat, phonons.
- Only **EPW's e-ph** breaks, and only where the ultrasoft / dense-grid path runs
  = **q≠Γ**. Γ is the reference q → the one that survives.
- The `gmap_sym` patch had no effect because the pseudo type, not that routine,
  is the cause.

---

## The fix

No epw-only shortcut — the pseudos are baked in from the SCF up. **This is a full
A0 redo.**

1. Get **norm-conserving** Li + F pseudos — Pseudo Dojo ONCV (PBE, standard or
   stringent) is the usual EPW choice; SSSP-precision also lists NC options.
2. Update `scf.in` / `nscf.in` / `ph.in`: new pseudo filenames; **lower `ecutrho`
   to ~4×`ecutwfc`** (NC doesn't need 8×); likely raise `ecutwfc` to ~60–80 Ry for
   ONCV and re-converge. The smaller dense grid is a bonus — it also removes the
   G-mapping fragility.
3. Rerun the chain: `nu run_pipeline.nu --from relax` (lattice constant may shift
   with new pseudos, so redoing vc-relax is cleanest) or `--from scf` to keep the
   geometry.

`run_pipeline.nu`'s `do-epw` finite-checks `epmatwp` and dies loudly on NaN, so a
still-broken run won't silently ship garbage.

---

## Dead ends (don't repeat these)

1. **Widen `dis_win_max` 24.50 → 66** ("enclose all bands so none is left
   uninitialised"). Made the coarse `epb` *worse* (8% → 99% NaN). Reverted. The
   out-of-window-bands theory was wrong.
2. **`gmap_sym` bounds patch** (`epw-gmap-sym-bounds.patch`). Rebuilt cleanly
   (patch listed in the QE derivation; store path changed `r55v0qws…`→`hj2nzsnp…`)
   yet output was **byte-identical** — still finite only at Γ. So `gmap_sym` OOB
   was not the cause. (The patch is a legitimate guard for the *crash* it was
   written for, just not this NaN.)
3. **"epb has real data, pivot to it."** False alarm from a *misaligned* `epb`
   parse (patt126's reader assumes the wrong band count — see Appendix). The real
   coupling block is 99.5% NaN too.

---

## Appendix A — `.epb` binary record layout (byte-verified)

patt126's `read_epb_mixed_record` does **not** match this file (it assumes one band
count for everything and leaves "padding"). The actual single Fortran record is:

| Field | Type | Shape (Fortran order) | Notes |
|---|---|---|---|
| `nqc` | int32 | scalar | = 216 (full BZ) |
| `xqc` | f64 | `(3, nqc)` | q-points, crystal coords; `xqc[:,0]` = Γ |
| `et` | f64 | `(20, nkc)` | **all 20 nscf bands** at the pool-local k; CBM = **index 5** (KS band 6, 9.463 eV) |
| `dynq` | complex128 | `(6, 6, nqc)` | dynamical matrix — **finite/correct** |
| `epmatq` | complex128 | `(15, 15, 6, nkc, nqc)` | e-ph; only the **first 5×5 band block is populated** (rest exact zero), and that block is ~all NaN except q=Γ |
| `zstar,epsi` | f64 | 27 reals (216 B) | trailing |

For `nkc = 22`, `nqc = 216`: total = 102 776 540 B payload (verified exact). Note
`et` uses 20 bands but `epmatq` a compacted 15 — the band index that is the CBM in
`et` (5) is **not** the CBM index inside `epmatq`'s block. k=Γ = pool-local index 0
(CBM eigenvalue is minimal there).

## Appendix B — reproduce the checks

```python
import numpy as np
from scipy.io import FortranFile
NBET, NBEP, NM, NK = 20, 15, 6, 22
f = FortranFile('out/LiF.epb1', 'r'); raw = f.read_record(dtype=np.byte); f.close(); o = [0]
def rd(dt, c):
    it = np.dtype(dt).itemsize; b = raw[o[0]:o[0]+c*it]; o[0] += c*it; return np.frombuffer(b, dtype=dt)
nqc = int(rd(np.int32, 1)[0]); xqc = rd(np.float64, 3*nqc).reshape((3, nqc), order='F')
et  = rd(np.float64, NBET*NK).reshape((NBET, NK), order='F')
dynq = rd(np.complex128, NM*NM*nqc).reshape((NM, NM, nqc), order='F')
ep  = rd(np.complex128, NBEP*NBEP*NM*NK*nqc).reshape((NBEP, NBEP, NM, NK, nqc), order='F')
blk = ep[:4, :4]                                   # the real coupling block
print("q with finite coupling:", [q for q in range(nqc) if np.isfinite(blk[..., q]).all()])  # -> [0] only
```

```bash
# epmatwp is uniform uninitialised garbage:
python3 -c "import numpy as np; w=np.fromfile('out/LiF.epmatwp',dtype=np.uint64,count=3_000_000); \
print('distinct 8-byte words:', len(np.unique(w)))"   # -> 1
```
