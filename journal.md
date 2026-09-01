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
