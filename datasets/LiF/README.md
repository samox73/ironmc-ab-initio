# LiF QE + EPW run (A0 for rmc-ab-initio)

Upstream DFT/DFPT/Wannier/EPW inputs that produce the compressed-tensor source for the
`rmc-ab-initio` DiagMC prototype. EPW's coarse Wannier-representation outputs
(`epwdata.fmt`, `LiF.epmatwp`, `LiF.epb1`) are consumed directly by this repo's
readers (`IO.read_epwdata_fmt`, `IO.read_epmatwp`, and `LowRank_Analisys.ipynb`), then
compressed and handed to ironmc.

## Structure & pseudopotentials

- LiF, rocksalt (Fm-3m), `ibrav = 2` (primitive FCC), 2 atoms/cell: Li (0,0,0), F (½,½,½).
- `celldm(1) = 7.6912 bohr` (~4.07 Å, PBE starting guess) — **refine with `relax.in` first**.
- Pseudos: SSSP **efficiency** PBE v2 (GBRV ultrasoft, consistent family for both elements)
  from `/home/samox/studies/phd/datasets/sssp/mix-sssp-pbe-eff-lib-v2/library/`:
  `Li.us.pbe.z_3.uspp.gbrv.v1.4.upf`, `F.us.pbe.z_7.uspp.gbrv.v1.4.upf`.
- Cutoffs: SSSP-eff minimums are Li 35/140, F 45/158 Ry. Used **50/400 Ry** (dual 8) for
  phonon safety on ionic LiF. Precision library (F 75/600) is the accuracy upgrade if
  phonons look off.

## Run order

```bash
# 0. (recommended) relax the PBE lattice, then copy the new celldm(1) into
#    scf.in / nscf.in / epw-stage inputs.
pw.x -in relax.in > relax.out

# 1. SCF
pw.x -in scf.in > scf.out

# 2. Phonons + dvscf on the 6x6x6 q-grid (also dielectric tensor + Born charges)
ph.x -in ph.in > ph.out

# 3. Gather ph.x output into ./save for EPW (EPW ships bin/pp.py; answer prefix = LiF)
python3 $EPW_DIR/bin/pp.py

# 4. NSCF on the full uniform 6x6x6 k-grid (nosym keeps all 216 points)
pw.x -in nscf.in > nscf.out

# 5. EPW: Wannierize + write the coarse tensors
epw.x -in epw.in > epw.out
```

Requires QE ≥ 7.x with EPW (patt126's Si provenance used PWSCF/PHONON 7.1, EPW 5.5).
Use pools consistent across ph/nscf/epw. `nscf.in` uses `automatic 6 6 6 0 0 0` with
`nosym=.true.` to keep the full grid; the rigorous EPW alternative is an explicit
`K_POINTS crystal` list from `kmesh.pl 6 6 6`.

## Knobs that MUST be tuned against output (not guessable a priori)

1. **Lattice** — set `celldm(1)` in all inputs to the `relax.in` result.
2. **Wannierization** (`epw.in`) — `proj`, `nbndsub`, `dis_win_max`, `dis_froz_min/max`
   are placeholders targeting the conduction (electron-polaron) manifold. After NSCF,
   read the actual band energies and set the windows, then validate: EPW writes
   `LiF_band.dat` (Wannier-interpolated); compare it to a `bands.x` DFT band structure
   over L–Γ–X and adjust until the CBM region matches. A bad Wannier fit here silently
   corrupts every downstream `g(k,q)`.

To include the valence F-2p manifold too (fuller `g`, cross-check against the classic
polar coupling): drop `bands_skipped`, raise `nbndsub`, and add `proj(...) = 'F:p'`.

## Downstream

`prefix = 'LiF'` throughout, so EPW writes `LiF.epmatwp`, `LiF.epb1`, and `epwdata.fmt`.
Point the patt126 readers/notebooks at `dataset/LiF/` with `prefix = "LiF"` (the notebooks
currently hardcode `"silicon"`). Then the planned `export_compressed(...)` (plan A1)
serializes the SVD/Tucker factors for `rmc-ab-initio`.
