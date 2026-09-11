# Phase B run configs (LiF)

Two validation stages. Run from the repo root, e.g.

```bash
cargo run --release -p rmc-ab-initio -- configs/stage1-frohlich-lif.json results-stage1
```

## The energy estimator needs manual calibration for α ≳ 1

The engine measures `E₀` with the exp-reweighted estimator
`⟨-exp((energy_estimate-mu)·τ)⟩` and, by default, **auto-updates**
`energy_estimate ← jackknife_energy` every `initial_self_consistent_period`
samples. That fixed-point map has slope ≈ −2.2 at strong coupling (measured), so
the naive iteration **diverges** for α ≳ 1 (blows up to ±1e17). It only stays
stable at the α=1 default.

So these configs **freeze** the estimate (`initial_self_consistent_period`
huge) and you calibrate three coupled knobs by hand:

1. **`mu` just below `E₀`** — not far below. `mu` must be under `E₀` (or `G(τ)`
   grows and diverges), but if it is *too* far under, the electron propagator
   `e^{-(−mu)τ}` confines the diagram to small τ and the phonon cloud is
   under-sampled → `mean_order` collapses and `E₀` comes out **too shallow**.
   Measured at α=4.76: `mu=-5.4` → `mean_order≈7–10` (good); `mu=-6.0` →
   `mean_order≈2` (starved, biased to −4.4 instead of −5.2). **Watch
   `mean_order`** — it should stay high (~7–10 at α≈5), not drop toward 1–2.
2. **`max_tau` scaled with coupling.** The estimator reweights by
   `exp((energy_estimate−mu)·τ)`; keep the product `(E₀−mu)·max_tau ≈ 2.5`
   (the α=1 default is `0.08·30`). At α=4.76 that means `max_tau ≈ 12`, not 30 —
   `max_tau=30` makes the reweight blow up (heavy tails, ±1–3 error, wild swings).
3. **Bisect `energy_estimate`** to the fixed point where `E ≈ energy_estimate`
   (~3–4 reruns). Use **`chains > 1`** for honest error bars.

`stage1` ships near the α=4.76 fixed point (`mu=-5.4`, `max_tau=12`,
`energy_estimate=-5.2`). Note a residual `max_tau`-dependence remains at strong
coupling, so a publication-grade `E₀(α_LiF)` still needs a small `max_tau→∞`
convergence study — the machinery itself is cleanly validated at α=2 below.

## Stage 1 — Fröhlich limit at α_LiF (vs Mishchenko 2000)

`|g|² ∝ α/q²` at α_LiF = 4.76 (envelope = target ⇒ reweight ≡ 1, i.e. the pure
Fröhlich engine). Verified fixed point **E₀ ≈ −5.2 … −5.3 ħω_LO ≈ −405 meV**,
consistent with Mishchenko's ≈ −5.2. (Sanity: α=2 → −2.09 vs literature −2.08.)

## Stage 2 — real LiF tensor (vs Sio–Giustino 2019)

First build the reduced-model cache once (needs `datasets/LiF/out/LiF.epmatwp`):

```bash
cargo run --release -p rmc-ab-initio -- model datasets/LiF datasets/LiF/reduced-model.json 8
```

then run `stage2-tensor-lif.json` and calibrate `energy_estimate` the same way.
Its E₀ is the polaron formation energy to compare against Sio–Giustino. Note
`material.mass_eff` (0.9) is an estimate — set it to your DFT CBM `m*/m_e`; it
fixes both α_LiF and the q-unit bridge.
```
