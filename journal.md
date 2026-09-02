# Ab-initio DiagMC

## SVD-compression convergence

Running the nsv (number of singular vectors) convergence test for second order energies and analyzing the output

```nu
$ cargo run --release --example nsv_convergence -p rmc-ab-initio
$ cat out | lines | each {|i| $i | parse "max_rank={max_rank}, e2_full={e_second_order_full}, e2_short={e_second_order_short}" } | flatten | into int max_rank | into float e_second_order_full e_second_order_short | add-diff --name delta_full e_second_order_full | add-diff --name delta_short e_second_order_short
╭───┬──────────┬─────────────────────┬──────────────────────┬────────────┬─────────────╮
│ # │ max_rank │ e_second_order_full │ e_second_order_short │ delta_full │ delta_short │
├───┼──────────┼─────────────────────┼──────────────────────┼────────────┼─────────────┤
│ 0 │       10 │              -10.79 │                -1.75 │          0 │           0 │
│ 1 │       25 │               -9.64 │                -1.34 │       1.14 │        0.41 │
│ 2 │       50 │              -10.73 │                -1.59 │      -1.09 │       -0.25 │
│ 3 │      100 │              -11.06 │                -1.70 │      -0.32 │       -0.12 │
│ 4 │      150 │              -11.13 │                -1.71 │      -0.07 │       -0.01 │
│ 5 │      200 │              -11.15 │                -1.69 │      -0.02 │        0.02 │
│ 6 │      300 │              -11.16 │                -1.68 │      -0.01 │        0.01 │
╰───┴──────────┴─────────────────────┴──────────────────────┴────────────┴─────────────╯
```

we see a convergence at around rank 150-200.

## Plan A implementation checks (2026-09-01)

- LiF fmt-only Fröhlich oracle: ω_TO(physical) = 33.568 meV,
  ω_TO(split-free) = 49.437 meV, ω_LO = 77.216 meV. The ratios
  `|g_LO|²/|g_F|²` at q = 1/32, 1/16, 1/8 are 0.994758, 0.979046,
  0.916563.
- 8³ second order with the 1 meV soft-mode cutoff and rank 200: six axial
  nearest-q LO terms = -0.025187 eV; full short+long = -0.104636 eV.
  Long-range-only is -0.027270 / -0.092457 eV respectively. The cutoff removes
  the old multi-eV acoustic-noise failure, but the plan's [-0.7,-0.2] gate is
  inconsistent with the independently evaluated 8³ LiF dispersion and remains
  a user checkpoint rather than a tuned assertion.
- HolsteinFlatModel (γ=0.6, n=4, max_tau=20, mu=-0.6, 4 chains,
  10⁶ steps, 10⁵ warm-up): fit = -0.36307 ± 0.01287, variable-τ thermo =
  -0.36271 ± 0.00409, fixed-τ thermo = -0.35719 ± 0.00121 (exact -0.36).

## A0 checkpoint resolution: the 8³ E² = −0.105 eV is correct (2026-09-01)

Two diagnostics settle the open checkpoint question:

- `examples/phonon_census.rs` (fmt-only): the 8³/12³/16³ q-grids contain **zero** modes below 1 meV (minima 8.70 / 5.87 / 4.42 meV) and no negative frequencies. Therefore the 1 meV cutoff never fires on these grids and **cannot** be what removed the −11 eV — that failure was fixed by the A0 refactor of the projection/interpolation chain itself (`reconstruct.rs`/`longrange.rs`/table build). The cutoff stays as dormant protection for finer grids, matching FEP-DMC's `phfreq_cutoff`.
- `examples/e2_ngrid.rs` (table-free, long-range-only E², reproduces the table value −0.092457 at 8³ exactly):

  | n | 1/n | e2_long [eV] |
  |---|-----|--------------|
  | 8 | 0.1250 | −0.092457 |
  | 10 | 0.1000 | −0.112330 |
  | 12 | 0.0833 | −0.130159 |
  | 14 | 0.0714 | −0.146093 |
  | 16 | 0.0625 | −0.160273 |
  | 20 | 0.0500 | −0.184024 |
  | 24 | 0.0417 | −0.202719 |
  | 32 | 0.0312 | −0.229412 |

  Monotone in 1/n with steepening slope, extrapolating to ≈ −0.31 eV from the last two points and heading toward the continuum weak-coupling limit −α·ω_LO ≈ −0.37 eV. The small 8³ value is pure grid suppression of the 1/q² Fröhlich weight (the plan's original [−0.7, −0.2] gate was calibrated against the continuum and is retired; see plan-lif-e-energy.md A0.4). Cross-checks: analytic six-axial-point estimate ≈ −0.031 eV vs code −0.0252 eV (m* uncertainty); FEP-DMC all-orders binding at 20³ is −0.248 eV vs our second-order ≈ −0.19 eV there — all-orders deepening of ~30% is plausible at α ≈ 4.8.

A0 is closed. Next: A4 ladder (8³ DMC smoke first; expect E₀ slightly below −0.105 eV), then Plans B/C/D.

## A4 step 1: 8³ DMC smoke test (2026-09-01)

`dmc_nsv_convergence` scout, 2M steps × 10 chains per rank, fixed-τ thermodynamic estimator, max_tau = 60 eV⁻¹, max_order = 500, 1 meV cutoff, 8³ grid:

| rank_cap | E0 [eV] | sign | mean order |
|---|---|---|---|
| 100 | −0.1104 ± 0.0010 | 0.994 | 7.06 |
| 150 | −0.1114 ± 0.0009 | 0.992 | 7.22 |
| 200 | −0.1129 ± 0.0009 | 0.993 | 7.29 |

All three checks pass: E₀ sits 6–8% below the same-grid E⁽²⁾ = −0.1046 eV (the expected all-orders deepening at this grid-suppressed coupling), ⟨sign⟩ ≈ 0.99 (single-band LiF(e) is nearly sign-free, matching the paper), mean order ≈ 7 ≪ max_order. Rank drift 100→200 is ~2 meV — rank 150–200 is adequate at 8³, to be re-checked per Plan B4.

Throughput: ~45 min wall for the sweep ⇒ ~0.7–2k steps/s/chain with the pre-Plan-C engine (full weight recompute per attempt, O(order·rank)). The 100M-step production ladder is blocked on Plan C (C1+C2).

## Review findings fixed (2026-09-02)

All 10 findings from the Plan A/C adversarial review are addressed; workspace at 226 passed / 0 failed, release-mode rmc-diagmc suite included.

- measure.rs: `DiagConfig::validate` now rejects `tau_fit > max_tau·(1−1/num_bins)` with a targeted message (the fixed-τ NaN foot-gun); a non-finite thermodynamic mean is a hard `Err` in fixed-τ mode and a `log::warn` otherwise; degenerate leave-one-out replicates (no order-0 samples, empty fit window) are skipped with a warning instead of aborting the run, with a hard floor of two valid replicates for any reported stderr; dropped-replicate counts are logged so the survivor-count rescaling is never silent.
- run.rs `DriftCheckedKernel`: the periodic check now also runs the full structural audit (`Diagram::check_consistency`, now compiled in release and public) — the only momentum-re-derivation guard that survives production builds — and the log-modulus drift assert uses a relative tolerance `1e-8·(1+|log W|)`.
- diagram.rs: new shared `segments()`/`arcs()` iterators are the single source of the walk conventions; `weight::log_modulus_and_phase` (which also drops the per-call `arc_keys()` Vec allocation) and `measure::local_energy` both use them.
- updates.rs: the truncated-exponential τ₂ draw and its reverse-move pdf now call rmc-core's `exponential_sample_bounded`/`exponential_pdf_bounded`; those helpers were upgraded to the `exp_m1`/`ln_1p` forms (bit-identical to the previous inline code, better than the old rmc-core forms at small λ·span), so rmc-frohlich benefits too.
- vertex_table.rs: the band edge is now the global minimum of `electron_ev` (warns if the CBM is off-Γ instead of silently assuming Γ); the hot path computes `kp` via integer `wrap_add(k,q)` (unit-tested against the old float round-trip on a full 5³×5³ sweep) and the per-call bounds assert is `debug_assert`.
- model_abinitio.rs: `propose_phonon` returns the pdf straight from the sampled index (no point↔index float round-trip).
- longrange.rs: `g_long_branch` takes `phfreq_cutoff_ev` explicitly; the Fröhlich-oracle test and `e2_ngrid` diagnostic now pass `DEFAULT_PHFREQ_CUTOFF_EV`, so diagnostics and engine share one coupling definition.

Refuted (no change): zero-arc-factor force-accept (unreachable), τ-pdf INF acceptance (precision-bounded + NaN-reject), ln(N_q) measure (proved correct), legacy-cache cutoff and double-precontraction (no such files can exist; revisit with a version field when Plan D's binary cache lands).
