//! LiF's small-q polar vertex against the analytic Fröhlich limit.

use nalgebra::Vector3;
use rmc_ab_initio::eph_svd::EphSvd;
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::longrange::{g_long_branch, gg_cart};
use rmc_ab_initio::model::lo_to_frequencies;
use rmc_ab_initio::reconstruct::{interpolate_d, interpolate_d_polar, interpolate_h};
use rmc_ab_initio::vertex_table::{
    TablePrecision, VertexTable, DEFAULT_MAX_BYTES, DEFAULT_PHFREQ_CUTOFF_EV,
};
use std::path::PathBuf;

const E2: f64 = 2.0;
const RY_TO_EV: f64 = 13.605698066;

fn lif(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/LiF")
        .join(rel)
}

#[test]
#[ignore = "512-point fmt-only A0 diagnostic"]
fn long_range_second_order_baseline() {
    let epw = EpwData::read(lif("epwdata.fmt")).unwrap();
    let crystal = Crystal::read(lif("crystal.fmt")).unwrap();
    let wigner = Wigner::read(lif("wigner.fmt")).unwrap();
    let cbm = interpolate_h(&epw, &wigner, Vector3::zeros()).0[0];
    let n = 8_usize;
    let mut all = Vec::new();
    let mut near = 0.0;

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                if i == 0 && j == 0 && k == 0 {
                    continue;
                }
                let q = Vector3::new(
                    i as f64 / n as f64,
                    j as f64 / n as f64,
                    k as f64 / n as f64,
                );
                let electron = interpolate_h(&epw, &wigner, -q).0[0] - cbm;
                let phonons = interpolate_d_polar(&epw, &crystal, &wigner, q);
                let branch = g_long_branch(&crystal, &epw, &wigner, q, DEFAULT_PHFREQ_CUTOFF_EV);
                for nu in 0..crystal.nmodes {
                    let omega = phonons.omega_thz[nu].abs() / 3289.84498 * RY_TO_EV;
                    let contribution =
                        branch[nu].norm_sqr() / (-electron - omega) / n.pow(3) as f64;
                    if [i, j, k]
                        .into_iter()
                        .filter(|&x| x == 1 || x == n - 1)
                        .count()
                        == 1
                        && [i, j, k].into_iter().filter(|&x| x == 0).count() == 2
                    {
                        near += contribution;
                    }
                    all.push((contribution, q, nu));
                }
            }
        }
    }

    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (contribution, q, nu) in all.iter().take(20) {
        eprintln!("q={q:?} nu={nu} contribution={contribution:.6} eV");
    }
    let total = all.iter().map(|entry| entry.0).sum::<f64>();
    eprintln!("E2_near={near:.6} eV E2_long={total:.6} eV");
    assert!((-0.05..=-0.01).contains(&near), "E2_near={near}");
    assert!((-0.15..=-0.05).contains(&total), "E2_long={total}");
}

#[test]
fn polar_lo_vertex_has_the_frohlich_magnitude() {
    let epw = EpwData::read(lif("epwdata.fmt")).unwrap();
    let crystal = Crystal::read(lif("crystal.fmt")).unwrap();
    let wigner = Wigner::read(lif("wigner.fmt")).unwrap();
    let (omega_to, omega_lo) = lo_to_frequencies(&epw, &crystal, &wigner);
    let eps_inf = epw.epsi[0][0];
    let eps_0 = eps_inf * (omega_lo / omega_to).powi(2);

    eprintln!(
        "omega_TO={:.6} eV omega_LO={:.6} eV",
        omega_to * RY_TO_EV,
        omega_lo * RY_TO_EV
    );
    assert!((omega_lo * RY_TO_EV - 0.082).abs() < 0.008);
    let split_free_to = interpolate_d(&epw, &crystal, &wigner, Vector3::zeros())
        .omega_thz
        .into_iter()
        .fold(f64::MIN, f64::max)
        / 3289.84498;
    assert!((split_free_to * RY_TO_EV - 0.050).abs() < 0.002);

    let mut ratios = Vec::new();
    for denominator in [32.0, 16.0, 8.0] {
        let q = Vector3::new(1.0 / denominator, 0.0, 0.0);
        let q_cart = gg_cart(&crystal, [0, 0, 0], q).norm();
        let analytic = (4.0 * std::f64::consts::PI * E2 / crystal.omega)
            * (omega_lo / 2.0)
            * (1.0 / eps_inf - 1.0 / eps_0)
            / q_cart.powi(2);
        let tabulated = g_long_branch(&crystal, &epw, &wigner, q, DEFAULT_PHFREQ_CUTOFF_EV)
            .into_iter()
            .map(|g| g.norm_sqr())
            .fold(0.0, f64::max)
            / RY_TO_EV.powi(2);
        let ratio = tabulated / analytic;
        eprintln!("q=1/{denominator:.0}: |g_LO|²/|g_F|²={ratio:.6}");
        ratios.push(ratio);
    }
    assert!((0.7..=1.3).contains(&ratios[0]), "ratios={ratios:?}");
}

#[test]
#[ignore = "full LiF SVD and 8³ A0 checkpoint"]
fn full_second_order_energy_is_physical() {
    let path = lif("out/LiF.epmatwp");
    if !path.exists() {
        eprintln!("skipping: epmatwp absent (regenerate via run_pipeline.nu)");
        return;
    }
    let epw = EpwData::read(lif("epwdata.fmt")).unwrap();
    let crystal = Crystal::read(lif("crystal.fmt")).unwrap();
    let wigner = Wigner::read(lif("wigner.fmt")).unwrap();
    let epmat = EpmatWp::read(&path, epw.nbnd, epw.nrr_k, epw.nmodes, epw.nrr_g).unwrap();
    if !epmat.data[0].is_finite() {
        eprintln!("skipping: epmatwp is non-finite");
        return;
    }
    let svd = EphSvd::compress(&epmat, 1.0e-8, 200);
    let table = VertexTable::build(
        &epw,
        &crystal,
        &wigner,
        &svd,
        8,
        DEFAULT_PHFREQ_CUTOFF_EV,
        TablePrecision::F64,
        DEFAULT_MAX_BYTES,
    )
    .unwrap();
    let k0 = table.index_of(Vector3::zeros()).unwrap();
    let mut contributions = Vec::new();
    for qi in 1..table.len() {
        let q = table.point(qi);
        let neg_q = table.index_of(-q).unwrap();
        for nu in 0..table.nmodes {
            let denominator = -table.electron_ev[neg_q] - table.phonon_ev[qi * table.nmodes + nu];
            let value =
                table.full_indices(k0, qi, nu).norm_sqr() / denominator / table.len() as f64;
            contributions.push((value, qi, nu));
        }
    }
    contributions.sort_by(|a, b| a.0.total_cmp(&b.0));
    for &(value, qi, nu) in contributions.iter().take(20) {
        eprintln!("q={:?} nu={nu} contribution={value:.6} eV", table.point(qi));
    }
    let total = contributions.iter().map(|entry| entry.0).sum::<f64>();
    let near = [
        Vector3::new(0.125, 0.0, 0.0),
        Vector3::new(-0.125, 0.0, 0.0),
        Vector3::new(0.0, 0.125, 0.0),
        Vector3::new(0.0, -0.125, 0.0),
        Vector3::new(0.0, 0.0, 0.125),
        Vector3::new(0.0, 0.0, -0.125),
    ]
    .into_iter()
    .map(|q| {
        let qi = table.index_of(q).unwrap();
        let nu = (0..table.nmodes)
            .max_by(|&a, &b| {
                table.long_ev[qi * table.nmodes + a]
                    .norm_sqr()
                    .total_cmp(&table.long_ev[qi * table.nmodes + b].norm_sqr())
            })
            .unwrap();
        contributions
            .iter()
            .find(|entry| entry.1 == qi && entry.2 == nu)
            .unwrap()
            .0
    })
    .sum::<f64>();
    eprintln!("E2_near={near:.6} eV E2_full={total:.6} eV");
    assert!((-0.05..=-0.01).contains(&near), "E2_near={near}");
    assert!((-0.2..=-0.05).contains(&total), "E2_full={total}");
}
