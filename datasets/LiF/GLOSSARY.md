# LiF QE + EPW cheat sheet

Every acronym / abbreviation used in this directory's inputs, scripts, and notes,
grouped into **workflow programs**, **input/output parameters**, and **general
(physics) terminology**. Purely-software terms (`free()`, pointers, MPI plumbing…)
are omitted.

The pipeline: **relax → scf → ph → pp → nscf → epw** turns a crystal structure into
the compressed electron–phonon tensors the `rmc-ab-initio` DiagMC prototype consumes.

---

## 1 · Workflow programs & stages

| Name | Expansion | What it does |
|------|-----------|--------------|
| **QE** | Quantum ESPRESSO | The DFT/DFPT suite these inputs drive. |
| **pw.x** | Plane-Wave (self-consistent field) | Core DFT engine; runs `relax`, `scf`, `nscf`, `bands`. |
| **relax** | (`vc-relax`) Variable-Cell RELAXation | pw.x run that relaxes atoms **and** the cell to minimise energy/stress → equilibrium lattice constant. |
| **scf** | Self-Consistent Field | Solves Kohn–Sham self-consistently on a reduced k-grid → ground-state density & potential. |
| **ph / ph.x** | PHonon | DFPT code: phonons (dynamical matrices) + (with `epsil`) dielectric tensor & Born charges; writes `dvscf` for EPW. |
| **pp** | Post-Processing gather (`epw_pp.py`, formerly `pp.py`) | Collects ph.x's scattered `dvscf`/dyn/pattern files into `./save/` in EPW's expected layout. |
| **nscf** | Non-Self-Consistent Field | pw.x on the **full** uniform k-grid using the fixed scf potential → Bloch states for Wannierisation. |
| **epw / epw.x** | Electron-Phonon Wannier | Wannier-interpolates the e-ph coupling; here writes the coarse tensors (`epwdata.fmt`, `.epmatwp`, `.epb1`). |
| **Wannier90** | — | The maximally-localised-Wannier-function library EPW calls internally. |
| **bands.x** | — | QE band-structure post-processor (produces the DFT reference band for the Wannier-fit check). |
| **kmesh.pl** | — | EPW helper that prints a Γ-centred uniform k-mesh as an explicit `K_POINTS crystal` list. |
| **DiagMC** | Diagrammatic Monte Carlo | The downstream polaron solver (`ironmc`/`rmc-ab-initio`) that eats the compressed tensors. |

---

## 2 · Input / output parameters

### pw.x — `&control`, `&system`, `&electrons`, `&ions`, `&cell`

| Parameter | Expansion / meaning |
|-----------|---------------------|
| **calculation** | Run type: `scf`, `nscf`, `vc-relax`, `bands`. |
| **ibrav** | BRAVais-lattice index (`2` = face-centred cubic). |
| **celldm(1)** | First CELL DiMension = lattice constant *a* (in bohr) for that `ibrav`. |
| **nat** | Number of AToms per cell. |
| **ntyp** | Number of atom TYPes (species). |
| **ecutwfc** | kinetic-Energy CUToff for WaveFunCtions (Ry) — plane-wave basis size. |
| **ecutrho** | energy CUToff for the charge density (ρ = "rho") (Ry); ultrasoft needs ≈ 8–12× `ecutwfc`. |
| **occupations** | Band-filling scheme (`fixed` = insulator, no smearing). |
| **nbnd** | Number of BaNDs to compute. |
| **nosym / noinv** | Disable crystal SYMmetry / spatial INVersion (affect k-grid folding; **must stay OFF for EPW here**). |
| **conv_thr** | SCF CONVergence THReshold (on total energy). |
| **mixing_beta** | Charge-density mixing factor between SCF iterations. |
| **diagonalization** | Eigensolver (`david` = Davidson). |
| **tprnfor / tstress** | Print FORces / compute STRESS tensor. |
| **forc_conv_thr / etot_conv_thr** | Relax stopping thresholds on FORCe / total ENergy. |
| **ion_dynamics / cell_dynamics** | Relaxation algorithm (`bfgs`). |
| **press / press_conv_thr** | Target PRESSure / its convergence threshold (vc-relax). |
| **prefix** | Filename stem for all QE I/O (`LiF`). |
| **outdir** | OUTput DIRectory for scratch/save files. |
| **pseudo_dir** | Directory holding the PSEUDOpotential files. |
| **restart_mode** | `from_scratch` vs restart from a previous run. |
| **verbosity** | Output detail level (`high`). |
| **amass** | Atomic MASS (a.m.u.) per species. |
| **K_POINTS** | k-mesh block: `automatic N N N s s s` (Monkhorst–Pack) or `crystal` (explicit fractional list). |
| **ATOMIC_SPECIES / ATOMIC_POSITIONS / CELL_PARAMETERS** | Species+pseudo table / atomic coordinates / lattice vectors. |

### ph.x — `&inputph`

| Parameter | Expansion / meaning |
|-----------|---------------------|
| **ldisp** | Logical: compute phonons over a whole DISPersion grid of **q**. |
| **nq1/nq2/nq3** | Phonon **q**-grid dimensions (6×6×6 here). |
| **epsil** | Compute the electronic dielectric tensor (EPSILon) + Born charges at Γ. |
| **tr2_ph** | Self-consistency THReshold² for the PHonon (DFPT) linear response. |
| **fildyn** | FILe for the DYNamical matrices (`LiF.dyn*`). |
| **fildvscf** | FILe stem for the `dvscf` output. |
| **dvscf** | Derivative of the self-consistent potential (δV_SCF/δu) wrt atomic displacement = the e-ph perturbation EPW needs. |

### epw.x — `&inputepw`

| Parameter | Expansion / meaning |
|-----------|---------------------|
| **elph** | Compute the ELectron-PHonon coupling. |
| **epwwrite / epwread** | WRITE / READ EPW's Wannier data (`epwdata.fmt`, `.epmatwp`). |
| **epbwrite / epbread** | WRITE / READ the coarse Bloch e-ph file (`.epb`). |
| **etf_mem** | Electron-Transport Fine-grid MEMory-handling mode. |
| **use_ws** | Use the Wigner–Seitz cell for the real-space R-vectors in the interpolation. |
| **lpolar** | Logical: treat the POLAR (Fröhlich long-range) part of the e-ph coupling using Born charges + ε∞. |
| **wannierize** | Run the Wannier90 construction step inside EPW. |
| **nbndsub** | Number of BaNDs in the Wannier SUBspace = number of Wannier functions. |
| **bands_skipped / exclude_bands** | Bands left out of Wannierisation (e.g. `1:5` = the 5 valence bands). |
| **num_iter** | Number of Wannier90 minimisation ITERations. |
| **dis_win_max** | DISentanglement outer WINdow MAXimum energy (states above are excluded). |
| **dis_froz_min / dis_froz_max** | DISentanglement FROZen (inner) window; bands inside are reproduced **exactly**. |
| **proj** | Wannier initial PROJections / trial orbitals (`Li:s`, `Li:p`). |
| **wdata** | Pass-through lines injected into the Wannier90 input (`wdata`). |
| **dvscf_dir** | Directory holding the gathered `dvscf` (`./save`). |
| **nk1..3 / nq1..3** | Coarse k-/q-grids (must match nscf / ph). |
| **nkf1..3 / nqf1..3** | Fine interpolation k-/q-grids. |
| **bands_plot / kpoint_path / bands_num_points** | Wannier90 band-interpolation controls (produce `LiF_band.dat`). |
| **-npool / -nk** | Split MPI ranks into k-point POOLs (for EPW, must equal the total rank count). |

### Files produced / consumed

| Name | Expansion / meaning |
|------|---------------------|
| **.upf** | Unified Pseudopotential Format (pseudopotential file). |
| **.save / data-file-schema.xml** | QE ground-state save directory / its metadata file. |
| **.wfc** | WaveFunCtion files. |
| **_ph0 · .phsave · patterns.xml · .dyn** | ph.x scratch: displacement-pattern & dynamical-matrix files. |
| **epwdata.fmt** | EPW master file: Wannier Hamiltonian *H(R)*, dynamical matrix *D(R)*, dimensions. |
| **.epmatwp** | Electron-Phonon MATrix in the Wannier (real-space) representation — "**w**annier electron + phonon **p**erturbation". |
| **.epb / .epb1** | Electron-Phonon coupling in the **B**loch representation on the coarse grid. |
| **LiF_band.dat** | Wannier-interpolated band structure, for the fit check vs a DFT band. |

---

## 3 · General terminology (physics)

### Method & pseudopotentials
- **DFT** — Density Functional Theory (the ground-state electronic-structure method).
- **DFPT** — Density Functional Perturbation Theory (linear response → phonons, e-ph, dielectric response).
- **XC** — eXchange-Correlation (the approximated part of the energy functional).
- **PBE** — Perdew–Burke–Ernzerhof, a widely-used GGA (Generalized Gradient Approximation) XC functional.
- **Pseudopotential** — replaces core electrons with an effective ionic potential so few plane waves suffice.
- **USPP / US** — UltraSoft PseudoPotential (Vanderbilt); softer wavefunctions but needs a high `ecutrho` + charge augmentation.
- **NC** — Norm-Conserving pseudopotential (harder, single density grid).
- **PAW** — Projector Augmented-Wave method (an all-electron-accurate scheme).
- **GBRV** — Garrity–Bennett–Rabe–Vanderbilt ultrasoft pseudopotential library.
- **SSSP** — Standard Solid-State Pseudopotentials — a curated, tested library ("efficiency" vs "precision" sets).
- **dual** — the `ecutrho / ecutwfc` ratio (8 here).

### Units & lattice
- **Ry** — Rydberg energy (1 Ry ≈ 13.606 eV).
- **eV** — electron-volt.
- **bohr / a.u.** — Bohr radius (≈ 0.529 Å) = atomic unit of length / "atomic units".
- **Å** — Ångström (0.1 nm).
- **a.m.u.** — atomic mass unit.
- **alat** — the length unit QE reports in = `celldm(1)`.
- **FCC** — Face-Centred Cubic lattice.
- **rocksalt / Fm-3m** — the NaCl crystal structure / its space group (what LiF is).
- **primitive cell / f.u.** — smallest repeating cell / formula unit (LiF = 1 f.u. per primitive cell).

### Reciprocal space & bands
- **BZ / IBZ** — (Irreducible) Brillouin Zone — the reciprocal-space unit cell / its symmetry-reduced wedge.
- **MP** — Monkhorst–Pack (the standard uniform k-grid scheme).
- **Γ, L, X** — high-symmetry points of the FCC Brillouin zone (Γ = zone centre, the band-gap location in LiF).
- **k-point / q-point** — electron crystal momentum / phonon crystal momentum.
- **Bloch state** — the periodic-crystal electron wavefunction labelled by (band, **k**).
- **band / manifold** — a single electronic energy band / a group of them.
- **CBM / VBM** — Conduction-Band Minimum / Valence-Band Maximum.
- **gap** — the CBM − VBM energy gap (LiF ≈ 8.8 eV in PBE).
- **G-vector / Miller index** — a reciprocal-lattice vector / its integer components.
- **FFT** — Fast Fourier Transform (the real↔reciprocal grid).
- **coarse vs fine grid** — the DFT/DFPT grid vs the dense Wannier-interpolation grid.

### Phonons, polarons & coupling
- **phonon** — a quantised lattice vibration.
- **LO / TO** — Longitudinal / Transverse Optical phonon branch.
- **LO–TO splitting** — the LO-vs-TO frequency gap at Γ in polar crystals, from the long-range macroscopic electric field.
- **ω_LO / ω_TO** — LO / TO phonon frequencies (LiF ω_LO ≈ 76 meV).
- **dynamical matrix** — Fourier-transformed inter-atomic force constants; its eigenvalues give phonon frequencies.
- **Born effective charge (Z\*)** — the "dynamical" charge linking an atomic displacement to macroscopic polarisation (drives LO–TO & the polar coupling).
- **ε∞ / ε0** — high-frequency (electronic) / static dielectric constant (LiF ε∞ ≈ 2.04).
- **e-ph** — electron-phonon (coupling).
- **g(k,q)** — the electron-phonon coupling matrix element (electron **k** scatters by phonon **q**).
- **Fröhlich coupling / α** — dimensionless strength of the long-range polar e-ph interaction (LiF α ≈ 5).
- **polaron** — an electron dressed by the lattice polarisation it induces.
- **m\*** — effective mass (band curvature); here the polaron's renormalised mass.
- **ZPR** — Zero-Point Renormalisation (phonon-induced shift of the band edges).
- **SOC** — Spin-Orbit Coupling (checked by `pp`; absent for light-element LiF).

### Wannier functions
- **WF** — Wannier Function: a localised orbital from a unitary transform of the Bloch states.
- **Wannierisation** — the construction of WFs (via Wannier90).
- **disentanglement** — extracting a smooth *N*-dimensional subspace out of entangled bands (Souza–Marzari–Vanderbilt).
- **frozen window** — the energy range whose bands are kept **exactly** during disentanglement.
- **spread (Ų)** — a WF's localisation measure ⟨r²⟩ − ⟨r⟩²; small = well localised.
- **WS** — Wigner–Seitz cell — the symmetric primitive real-space cell used for the interpolation R-vectors.

### Downstream compression
- **SVD** — Singular Value Decomposition (low-rank matrix factorisation).
- **Tucker** — Tucker decomposition (the tensor generalisation of SVD).
- **compressed tensor** — the low-rank representation of *g* handed to the DiagMC solver.
