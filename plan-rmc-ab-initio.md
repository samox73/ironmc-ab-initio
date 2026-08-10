# Plan: rmc-ab-initio DiagMC prototype (LiF)

## Status & locations (updated 2026-08-10)

Everything now lives in **ironmc** (this repo), one coordinated location:
- **This plan** — `ironmc/plan-rmc-ab-initio.md`.
- **LiF QE/EPW dataset (Phase A0 outputs)** — `ironmc/datasets/LiF/`. The ~1.4 GB of
  binary EPW output (`out/`, `save/`, Wannier intermediates) is **gitignored** but
  present on disk; regenerate any time with `nu datasets/LiF/run_pipeline.nu`.
  Tracked in git (15 files, ~1.1 MB): the `*.in` decks, `*.nu` scripts,
  README/RUN/GLOSSARY/**EPH-NAN-DIAGNOSIS**, the EPW patch, and the **three** small files
  A1 parses (`epwdata.fmt`, `crystal.fmt`, `wigner.fmt`).
- **Phase-A export code (A1)** — written **fresh in Rust** in `crates/apps/rmc-ab-initio`
  (patt126's `IO.py`/`Wannier2Bloch.py` = reference only, not copied). Modules: `epw`
  (readers) / `reconstruct` (eigensolve + Γ reconstruction). Tests: `datasets_lif.rs`,
  `reconstruct_lif.rs`.

**A0 is DONE — but only after a full redo (see `datasets/LiF/EPH-NAN-DIAGNOSIS.md`).**
The original **ultrasoft** (SSSP GBRV) deck produced an all-NaN e-ph tensor: EPW's e-ph
requires **norm-conserving** pseudopotentials, and ultrasoft gave finite coupling only at
q=Γ, NaN everywhere else (→ 100% NaN `epmatwp`). Fixed by switching Li+F to **Pseudo Dojo
ONCV NC-SR-PBE** (`ecutwfc 50→96`, `ecutrho 400→384`; ~3–4× slower ph.x). Deliverables in
`datasets/LiF/`: `epwdata.fmt` (dims `nbnd=4,nrr_k=279,nmodes=6,nrr_q=401,nrr_g=340` +
Ef=0.0526 Ry + `Z*=±1.055` + `ε∞=2.044` + Wannier `H`/`D`), `out/LiF.epmatwp` (146 MB,
**finite**), `out/LiF.epb1..10` (coarse Bloch, **finite at all 216 q**). Wannier manifold
= 4 WFs (Li:s + Li:p, **CBM band-6 frozen** → CBM exact). Deck prereqs: explicit Γ-centred
`K_POINTS crystal` grid; one QE version across scf/ph/nscf/epw; `ulimit -s unlimited`;
disentanglement windows retuned to the NC band edges (CBM=9.614 eV, p-triplet top 24.380).
The `gmap_sym` patch turned out **irrelevant** to the NaN — kept as a harmless latent-bug
guard.

Next: **A1 item 8** (model output) + Phase B. Items 1–7 **and A2 DONE + validated**:
the A2 round-trip closes to **machine precision** — at all 215 nonzero coarse q,
`g_full(epb) = g_short(exact bra) + ⟨u_{Γ+q}|u_Γ⟩·g_L` to worst cos=1.000000,
|ratio−1|=3e−7 (`tests/epb_lif.rs`). New modules `longrange` (item 7) + `epb` (A2 reader);
`reconstruct` grew an open-bra tensor (`g_cart_exact`) so the round-trip resolves the
electron-leg k-dependence the reduced Γ-vertex drops. **Not yet committed.**

---

## Context

`ironmc` is a Rust DiagMC engine whose only physics app today is the **analytic
Fröhlich polaron** (`crates/apps/rmc-frohlich`). We want a **generic ab-initio
DiagMC** that swaps the analytic vertex for **real electron-phonon tensors** and
test it on **LiF** — a wide-gap ionic insulator that is nearly the textbook
Fröhlich case (single conduction-band minimum at Γ, one dominant LO branch,
Fröhlich coupling α ≈ 5), which makes it the cleanest first material and gives a
built-in analytic-limit check.

The tensors come from `patt126-ab-initio-diagmc`, an EPW post-processor. Two facts
from exploration shape this plan:

- **patt126 has no LiF data** (everything shipped is silicon) and **no on-disk
  compressed export**: its SVD/Tucker compression is in-memory only; the sole file
  it writes is a *dense* EPW `.epmatwp`. Its evaluate-g path is
  `Wannier2Bloch.fourierANDlr(U,Vt,s,q,k,R)` + `Wannier2Bloch_rotate`.
- **ironmc has no external-data path.** All Fröhlich physics is pure scalar leaves
  in `crates/apps/rmc-frohlich/src/physics.rs`; the engine (`Runner`,
  `Update<State>`, `MetropolisKernel`) is fully state-generic. `plan-gpu.md`
  Phase 5 already *names* a `Model` trait as the intended seam. Unused-but-present:
  `rmc-numeric::LinearInterpolationNd` + `rmc-grids::NdGrid` for tensor interpolation.

**Decisions (from user):** build the compressed export in patt126 first; house the
consumer in a **new `crates/apps/rmc-ab-initio`**; validate against **both**
Fröhlich DiagMC curves (Mishchenko 2000) **and** ab-initio LiF polaron numbers
(Sio–Giustino 2019).

## Prototype physics scope (affirming the user's framing)

- **Single band** — LiF CBM at Γ is non-degenerate & parabolic; the electron
  polaron only needs one band. (Hole/VBM is degenerate → harder; electron is the
  clean first case.) Because k stays near Γ, we approximate `g(k,q) ≈ g(Γ,q)`, so
  the vertex is a function of **q only**.
- **One dominant LO branch** — keep only the Fröhlich-active LO mode of LiF's 6
  branches. Dominant for the formation energy; other modes deferred.
- **Einstein phonon** — reuse the engine's constant-ω structure at ω_LO(Γ) (or a
  q-averaged ω_LO). Dispersive ω_LO(q) is a later one-line swap via the Model trait.
- **Direct compressed evaluation, reduced form** — we need only
  `q ↦ |g(Γ,q,LO)|²` (and ω_LO). Reconstruct it from the compressed factors on the
  q-grid **once**, then interpolate with `LinearInterpolationNd` per proposal
  (precompute-then-interpolate beats re-doing the Fourier sum every step).

## Phase A — LiF tensor reader + reconstruction, **fresh Rust in ironmc** [new, additive]

A0 produced the EPW files; A1 turns them into what the DiagMC needs. Written from
scratch in `crates/apps/rmc-ab-initio` — patt126's `IO.py`/`Wannier2Bloch.py` are a
**reference only, not copied** (and the rewrite is *more correct*, see item 4). Deps
are all already in the workspace: `num-complex`, `nalgebra`, `serde`/`serde_json`,
`rmc-numeric`. **No SVD/Tucker, no npz/HDF5, no LAPACK** — the reduced Γ-slice
`|g(Γ,q,ν)|²` is a scalar field (<1 MB on any grid), so the whole compression layer
is YAGNI here (revisit only if we ever store the full `g(k,q)` tensor).

Module split in `rmc-ab-initio`: `epw` (file readers) / `reconstruct` (Wannier
interpolation) / `longrange` (polar term) / `model` (the reduced artifact + B-side).

- **A0 — DONE** (norm-conserving redo — see status header + `EPH-NAN-DIAGNOSIS.md`).
  `epwdata.fmt`, `out/LiF.epmatwp` (finite), `out/LiF.epb*` (finite) in `datasets/LiF/`.

- **A1 — readers + reduced reconstruction. Implementation inventory (~450 ln):**
  1. **`epwdata.fmt` parser** (text, ~60 ln): line1 Ef; line2 dims
     `(nbnd=4, nrr_k=279, nmodes=6, nrr_q=401, nrr_g=340)`; line3 the **Z\*/ε∞ polar
     block** (Z*_Li=+1.05, Z*_F=−1.05, ε∞=2.04 — patt126 skips this); then complex
     `H(R_e)` [nrr_k,nbnd,nbnd] and `D(R_p)` [nrr_q,nmodes,nmodes].
  2. **`crystal.fmt` parser** (text, ~30 ln): cell volume Ω=113.45 bohr³, lattice
     `at`, masses, atomic positions τ.
  3. **`epmatwp` binary reader** (~30 ln): flat little-endian `(f64 re, f64 im)`
     stream → `Complex<f64>`, **no Fortran record markers**, Fortran-order strides
     `(nbnd,nbnd,nrr_k,nmodes,nrr_g)` → index as `g_wann(R_e, R_p, mode, n, m)`.
     Just `std::fs::read` + `f64::from_le_bytes`.
  4. **`wigner.fmt` reader** (~30 ln): the WS R-vectors **and degeneracy weights** for
     the three sets R_k(279) / R_q(401) / R_g(340). Use these, don't re-derive the WS
     scan. NB the sets DIFFER, so g_wann's R_e axis (nrr_k=279) and R_p axis
     (nrr_g=340) index *different* vector lists. (Correctness win: patt126's
     `read_epmatwp` wrongly assumes nrr_k==nrr_g when sorting.)
  5. **Tiny Hermitian eigensolve** for `H(Γ)` 4×4 and `D(q)` 6×6 → `nalgebra` via the
     real 2n×2n embedding of the complex-Hermitian matrix, or an ~80-line cyclic
     Jacobi. Tiny matrices, either is fine.
  6. **`reconstruct` at k=Γ** (~100 ln): the electron phase e^{iΓ·R_e}=1, so collapse
     `ḡ(R_p) = Σ_{R_e} g_wann(R_e,R_p)/deg(R_e)`; then
     `g(Γ,q,ν) = Σ_{R_p} (e^{iq·R_p}/deg(R_p)) · U†(Γ) ḡ(R_p) U(Γ) · e_ν(q)`, with
     `U(Γ)` from diag `Σ_R H(R)` and `e_ν(q), ω_ν(q)` from diag `D(q)`. Take the CBM
     band (lowest of the 4) and the LO branch.
  7. **`longrange` — the ONE hard piece (new physics, not in patt126).** The analytic
     Fröhlich term `g_L(q,ν)` that `lpolar=.true.` subtracted from `epmatwp`. Verdi–
     Giustino dipole, EPW `longrange.f90::rgd_blk_epw`:
     `g_L ∝ (i·4πe²/Ω) Σ_G [(q+G)·Z*_κ·e_ν,κ / ((q+G)·ε∞·(q+G))] · e^{−i(q+G)·τ_κ}`
     with an Ewald Gaussian cutoff `e^{−(q+G)·ε∞·(q+G)/(4·α_Ew)}`. Total
     `g = g_short(epmatwp) + g_L`. Match EPW's convention exactly — this restores
     LiF's dominant 1/q coupling. Validate hard in A2. **DONE** — `src/longrange.rs`
     ports `rgd_blk_epw` (3D dipole branch, `alph=1`, `gmax=14`, `e2=2`) in the
     **Cartesian atom-displacement basis** (`imode=3κ+α`, Ry), band-diagonal in the
     smooth basis (`[U_{k+q}U_k†]` deferred to the consumer — see A2). Returns the
     pure dipole term Phase B consumes.
  8. **`model` output** (~easy): reduced `|g(Γ,q,ν)|²` on a q-grid + `ω_ν(q)` + `ε(Γ)`
     + α_LiF constants. Feed the Phase-B `Model` directly, or a small `serde` cache to
     avoid re-reading the 146 MB `epmatwp` each run.

  **Status (2026-08-10): items 1–7 + A2 DONE + validated** against the real (finite,
  norm-conserving) `datasets/LiF` files — all `rmc-ab-initio` tests green: dims
  `4/279/6/401/340`, Ef=0.0526, ε∞=2.044, Z*=±1.055; H(Γ) CBM=9.614 eV below the 3-fold
  p-manifold (24.380); D(Γ) acoustic sum rule + TO=11.954 THz; reduced `|g(Γ,q,LO)|²`
  finite from the real tensor. Item 5 used the nalgebra real-2n×2n embedding +
  Gram-Schmidt (robust to LiF's degenerate p/TO manifolds). Item 7 + A2 land the
  `longrange` + `epb` modules and an open-bra tensor in `reconstruct`
  (`g_cart_exact`); the A2 round-trip is **exact to machine precision** (below).
  **Item 8 next, then Phase B.** One commit per landed group (`feat:`), per AGENTS.md —
  the crate is not yet committed.

- **A2 — round-trip validation. DONE + exact.** `src/epb.rs` reads one `.epb` pool file:
  a **single Fortran record** (4-byte leading+trailing length markers), layout
  `nqc(i4), xqc(3,nqc), et_loc(nbnd,nks), dynq(nm,nm,nqc), epmatq(nbndep,nbndep,nks,nm,nqc),
  zstar+epsi(27 reals)`, all LE f64/c16 col-major. **Corrections to the earlier
  byte-guess:** the mode axis of `epmatq` is `nks·nmodes` interior (not `nmodes,nkc`),
  `et_loc` is the full `nbnd=20` band set, `nbndep=15` (bands 6–20), and CBM = **kept-band
  local index 0** in `epmatq` (≠ `et` index 5). `nks` (pool-local k, 22 for epb1) is
  recovered from the record length. epb1 holds Γ at pool-local `ik=0`; we keep the ~100 MB
  `epmatq` as raw bytes and decode only the CBM slice on demand.
  **Result:** the epb is the FULL coupling (short+long), so
  `g_full = g_short(exact bra u_{Γ+q}) + ⟨u_{Γ+q}|u_Γ⟩·g_L` — the overlap
  `⟨u_{Γ+q}|u_Γ⟩` is EPW's `[U_{k+q}U_k†]_{CBM}` rotating the smooth-basis `g_L` into the
  epb's raw KS gauge. Over **all 215 nonzero coarse q** the round-trip closes to worst
  cos=1.000000, |ratio−1|=3e−7 (gauge-invariant: epb KS states carry an arbitrary per-k
  phase). Plus the Fröhlich **1/q** signature: `‖g_L‖` grows toward Γ. Proves readers +
  reconstruction + longrange together (`tests/epb_lif.rs`).

  **Note for Phase B:** the *reduced* Γ-vertex (both legs `u_Γ`, `⟨u|u⟩=1`) does **not**
  close the round-trip — it overshoots ‖·‖ by ~9% at the nearest shell because it can't
  resolve the CBM's k-dependent character (‖overlap‖ falls from 0.94 at the first shell to
  0 at the zone boundary). The exact bra + overlap is validation-only; whether Phase B needs
  it (vs the q-only reduced vertex) is a B2/B4 accuracy call.

## Phase B — `rmc-ab-initio` crate in ironmc (Rust) [new, additive]

Reuse the state-generic engine unchanged. The `rmc-ab-initio` crate already exists
from Phase A; Phase B grows the DiagMC consumer inside it.

- **B1 — single vertex-weight seam (revised after reading the real engine).**
  Key finding: in Fröhlich units (ħ = m\* = ω_LO = 1) the prototype's electron
  dispersion is *exactly* `p²/2` and the phonon energy is *exactly* `1`, so
  **`MASS`/`OMEGA` never differ from the existing consts** — the *only*
  material-specific quantity is the vertex `|g(q)|²`. Therefore B1 is **not** an
  engine-wide `Model`-trait/mass-omega parameterization (that would be a large
  bit-for-bit-risky diff across both mirrored engines for zero prototype benefit).
  B1 is one seam: a Fröhlich `vertex_weight_sq(q)` that the ab-initio path reweights
  by `|g_ab(q)|²/|g_F(q)|²`. Its clean boundary is defined by the B3 reweighting +
  B2 tensor evaluator, so **land it together with B2** — both need the LiF data.
  Guard bit-for-bit with the existing parity tests
  (`tests/flat_ratio_parity.rs`, `tests/flat_diagram.rs`).
- **B2 — `AbInitioModel`:** consume Phase-A's `reconstruct`/`model` output
  (in-process, or the small `serde` cache) — `|g(Γ,q,LO)|²` and `ω_LO(q)` on the
  q-grid — wrapped in `rmc-numeric::LinearInterpolationNd` over `rmc-grids::NdGrid`.
  With A1 done, B2 is mostly wiring.
- **B3 — vertex swap by reweighting (lazy path):** keep drawing q from the existing
  Fröhlich envelope (`draw_new_q_from_uniforms`) and multiply the acceptance by
  `|g_abinitio(q)|² / |g_envelope(q)|²`. Reuses the sampler; efficient because LiF's
  g *is* Fröhlich-like at small q. Leave the envelope scale as a **calibration
  knob** (importance-sampling efficiency tuning). Native q-sampler deferred.
- **B4 — unit bridge:** nondimensionalize to Fröhlich units (ħ = m\* = ω_LO = 1):
  compute α_LiF from (ε∞, ε0, ω_LO, m\*); convert q (Å⁻¹) → Fröhlich momentum and
  |g|² (eV²) → ħω_LO units; report E₀ back in meV via ħω_LO. This bridge is the
  fiddly step — validated directly by rung 1 below.
- **B5 — thin app + config:** a `RunConfig`-style JSON (mirror
  `rmc-frohlich/config.rs`) adding a tensor-file path + material constants; reuse
  `measurement.rs` jackknife estimators for E₀, Z; effective mass via a `momentum`
  sweep.

## Validation ladder (both targets the user picked)

0. **Refactor parity:** `FrohlichModel` in the new crate reproduces α=1 →
   `E≈-1.016`, `Z≈0.59` (reuse `rmc-frohlich` parity harness). Proves B1.
1. **Fröhlich-limit + unit-bridge check:** feed `AbInitioModel` the analytic
   LiF-Fröhlich vertex `|g(q)|² ∝ α_LiF/q²`; it must match the pure Fröhlich engine
   at α_LiF against **Mishchenko 2000** DiagMC `E₀(α)`, `m*(α)` + Feynman. Then
   check the real tensor's `|g(Γ,q→0,LO)|²` recovers the same 1/q² Fröhlich
   asymptote (standard ab-initio sanity check; validates B2+B4 together).
2. **Material comparison:** real LiF compressed g → polaron formation energy / ZPR
   and effective mass vs **Sio–Verdi–Poncé–Giustino 2019** ab-initio LiF polaron
   numbers.

## Verification (end-to-end)

- `rtk cargo test --workspace` green at the end of every phase (incl. the untouched
  Fröhlich parity tests — the refactor's guard rail).
- Phase A: run the A2 round-trip cell; assert reconstructed g matches dense
  `BENCHMARK` within the reported rank-10 SVD error (~2e-3 on Si).
- Phase B rung 0: `cargo run -p rmc-ab-initio -- <frohlich-limit config>` and diff
  E₀/Z against `-1.016 / 0.59`.
- Rung 1/2: run at α_LiF and at real-LiF tensors; compare printed E₀ (meV), Z, and
  the momentum-sweep m\* to the literature tables above.
- Benchmarks/GPU: **do not run** (AGENTS.md) — mark USER CHECKPOINT and hand the
  exact command to the user.

## Deferred (ponytail — add when needed)

Engine-wide `MASS`/`OMEGA` parameterization + full `Model`-trait genericization of
the eight updates (the Fröhlich unit system already fixes m\* = ω_LO = 1 for the
prototype; these are needed **only** for the non-parabolic-ε(k) / dispersive-ω_LO(q)
relaxations below). Multi-branch phonons; dispersive ω_LO(q); full `g(k,q)`
k-dependence (beyond Γ-slice); native |g_abinitio|² q-sampler; on-the-fly compressed
contraction (precompute+interpolate for now); GPU `#[cube]` path for the ab-initio
model. Hole polaron (degenerate VBM).

## Notes / constraints (AGENTS.md)

Prefix shell with `rtk`; commits `type: summary`, one phase-step per commit; new
code additive, only `refactor:`-style pure-function extraction may touch existing
files; leave `rmc-frohlich/input.json` alone; don't run benchmarks/GPU yourself.
