//! Rung 0 (plan validation ladder): with an identity reweight the ab-initio
//! consumer *is* the Fröhlich engine. Same seed, same update order/weights, the
//! reweight multiplies every arc-`q` move by exactly 1.0 ⇒ the two runs must be
//! **bit-for-bit** identical (and land on E≈-1.016, Z≈0.59). This is the guard
//! that reusing frohlich's physics through the reweight seam changed nothing.

use rmc_ab_initio::config::{AbInitioConfig, VertexSpec};
use rmc_ab_initio::vertex::Material;
use rmc_frohlich::config::RunConfig;

fn cfg() -> RunConfig {
    // Enough steps that the energy estimate is finite; parity holds at any length.
    RunConfig {
        max_steps: 300_000,
        ..RunConfig::default()
    }
}

#[test]
fn abinitio_identity_matches_frohlich_bit_for_bit() {
    let run = cfg();
    // Reference: the Fröhlich engine's own single-chain path (same seeding as ours).
    let bench = rmc_frohlich::app::run_bench(&run).unwrap();

    let ab = AbInitioConfig {
        run: run.clone(),
        material: Material {
            eps_inf: 2.0437,
            eps_zero: 9.0,
            omega_lo_mev: 77.2,
            mass_eff: 0.9,
        },
        vertex: VertexSpec::Frohlich { alpha: run.alpha }, // α_target = α_engine ⇒ R ≡ 1
        table_max_bytes: rmc_ab_initio::vertex_table::DEFAULT_MAX_BYTES,
    };
    let (report, _out) = rmc_ab_initio::run::run(&ab).unwrap();

    let e_ref = bench.summary.energy.mean;
    let z_ref = bench.summary.quasiparticle_weight.mean;
    assert!(
        e_ref.is_finite() && z_ref.is_finite(),
        "reference not finite"
    );

    // Bit-for-bit: the reweight seam is transparent at identity.
    assert_eq!(
        report.summary.energy.mean, e_ref,
        "energy diverged from Fröhlich"
    );
    assert_eq!(
        report.summary.quasiparticle_weight.mean, z_ref,
        "Z diverged from Fröhlich"
    );

    // And the physics target of the ladder.
    assert!(
        (e_ref - (-1.016)).abs() < 0.05,
        "E = {e_ref} (target ≈ -1.016)"
    );
    assert!((z_ref - 0.59).abs() < 0.05, "Z = {z_ref} (target ≈ 0.59)");

    // Unit bridge sanity: E in meV = E_Fröhlich · ħω_LO.
    assert!((report.energy_mev - e_ref * 77.2).abs() < 1e-9);
}
