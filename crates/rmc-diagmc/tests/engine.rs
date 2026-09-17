use std::sync::Arc;

use nalgebra::Vector3;
use num_complex::Complex64;
use rand::{Rng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rmc_diagmc::weight::{self, WeightRatio};
use rmc_diagmc::{run, DiagConfig, Diagram, FrohlichModel, Model};

struct ComplexModel;

impl Model for ComplexModel {
    fn n_branches(&self) -> usize {
        1
    }

    fn electron_energy(&self, k: Vector3<f64>) -> f64 {
        k.norm_squared() / 2.0
    }

    fn phonon_energy(&self, q: Vector3<f64>, _nu: usize) -> f64 {
        0.8 + 0.1 * q[0].abs()
    }

    fn vertex(&self, k: Vector3<f64>, q: Vector3<f64>, _nu: usize) -> Complex64 {
        Complex64::from_polar(1.0 + 0.1 * q.norm_squared(), k[0] + q[1])
    }

    fn propose_phonon(&self, _rng: &mut dyn RngCore) -> (Vector3<f64>, usize, f64) {
        (Vector3::new(0.2, 0.1, -0.1), 0, 1.0)
    }

    fn propose_phonon_pdf(&self, _q: Vector3<f64>, nu: usize) -> f64 {
        f64::from(nu == 0)
    }
}

#[derive(Clone, Copy)]
struct HolsteinFlatModel {
    n: usize,
    gamma: f64,
}

impl Model for HolsteinFlatModel {
    fn n_branches(&self) -> usize {
        1
    }

    fn electron_energy(&self, _k: Vector3<f64>) -> f64 {
        0.0
    }

    fn phonon_energy(&self, _q: Vector3<f64>, _nu: usize) -> f64 {
        1.0
    }

    fn vertex(&self, _k: Vector3<f64>, _q: Vector3<f64>, _nu: usize) -> Complex64 {
        Complex64::new(self.gamma, 0.0)
    }

    fn propose_phonon(&self, rng: &mut dyn RngCore) -> (Vector3<f64>, usize, f64) {
        let index = rng.gen_range(0..self.n.pow(3));
        let q = Vector3::new(
            (index / self.n.pow(2)) as f64 / self.n as f64,
            (index / self.n % self.n) as f64 / self.n as f64,
            (index % self.n) as f64 / self.n as f64,
        );
        (q, 0, self.propose_phonon_pdf(q, 0))
    }

    fn propose_phonon_pdf(&self, _q: Vector3<f64>, nu: usize) -> f64 {
        if nu == 0 {
            1.0 / self.n.pow(3) as f64
        } else {
            0.0
        }
    }

    fn log_momentum_measure(&self) -> f64 {
        (self.n.pow(3) as f64).ln()
    }
}

fn close_complex(a: Complex64, b: Complex64) {
    assert!((a - b).norm() < 1.0e-10, "{a} != {b}");
}

#[test]
fn incremental_bookkeeping_matches_full_weight_after_random_moves() {
    let model = ComplexModel;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(17);
    let mut diagram = Diagram::new(-1.0, Vector3::new(0.1, 0.0, 0.0), 8.0, 0, 8);
    let mut tracked_modulus = weight::weight_modulus(&diagram, &model);
    let mut tracked_phase = weight::phase(&diagram, &model);

    for _ in 0..200 {
        let before = diagram.clone();
        let ratio: WeightRatio = if (diagram.order == 0 || rng.gen_bool(0.6)) && diagram.order < 8 {
            let mut times = [
                rng.gen_range(0.01..diagram.tau()),
                rng.gen_range(0.01..diagram.tau()),
            ];
            times.sort_by(f64::total_cmp);
            if times[0] == times[1] {
                continue;
            }
            let q = Vector3::new(rng.gen_range(-0.5..0.5), rng.gen_range(-0.5..0.5), 0.2);
            diagram.insert_arc(times[0], times[1], q, 0);
            weight::add_phonon_ratio(&before, &diagram, &model)
        } else {
            let arcs = diagram.arc_keys();
            let (left, right) = arcs[rng.gen_range(0..arcs.len())];
            diagram.remove_arc(left, right);
            weight::remove_phonon_ratio(&before, &diagram, &model)
        };
        tracked_modulus *= ratio.modulus;
        tracked_phase *= ratio.phase;
        let exact_modulus = weight::weight_modulus(&diagram, &model);
        let scale = exact_modulus.abs().max(1.0);
        assert!((tracked_modulus - exact_modulus).abs() < 1.0e-10 * scale);
        close_complex(tracked_phase, weight::phase(&diagram, &model));
    }
}

#[test]
fn frohlich_run_is_exactly_sign_free() {
    let cfg = DiagConfig {
        mu: -1.2,
        max_tau: 8.0,
        tau_fit: 4.0,
        num_bins: 32,
        chains: 2,
        max_steps: 40_000,
        warmup_steps: 5_000,
        steps_per_cycle: 2,
        n_batches: 4,
        ..DiagConfig::default()
    };
    let report = run(Arc::new(FrohlichModel::new(1.0)), cfg).unwrap();
    assert_eq!(report.sign.mean, 1.0);
    assert_eq!(report.sign.stderr, 0.0);
}

#[test]
#[ignore = "statistical A1/A2 correctness run"]
fn holstein_flat_energy_is_grid_independent() {
    let target = -0.36;
    let base = DiagConfig {
        mu: -0.6,
        max_tau: 20.0,
        tau_fit: 10.0,
        num_bins: 40,
        chains: 4,
        max_steps: 1_000_000,
        warmup_steps: 100_000,
        steps_per_cycle: 5,
        n_batches: 20,
        ..DiagConfig::default()
    };
    let model = Arc::new(HolsteinFlatModel { n: 4, gamma: 0.6 });

    let report = run(model.clone(), base.clone()).unwrap();
    assert_within_3sigma("fit", report.e0.unwrap(), target);
    assert_within_3sigma("thermo", report.e0_thermo, target);

    let report = run(
        model,
        DiagConfig {
            fixed_tau: true,
            ..base
        },
    )
    .unwrap();
    assert!(report.e0.is_none() && report.z.is_none());
    assert_within_3sigma("fixed-tau thermo", report.e0_thermo, target);
}

fn assert_within_3sigma(name: &str, estimate: rmc_diagmc::Estimate, target: f64) {
    eprintln!("{name}: {} +/- {}", estimate.mean, estimate.stderr);
    assert!(
        (estimate.mean - target).abs() <= 3.0 * estimate.stderr,
        "{name}: {estimate:?}, target={target}"
    );
}

#[test]
#[ignore = "10M-step statistical correctness run; run explicitly in release mode"]
fn frohlich_mishchenko_stage_zero() {
    for (alpha, target) in [(1.0, -1.013), (2.0, -2.08)] {
        let cfg = DiagConfig {
            mu: target - 0.2,
            max_steps: 10_000_000,
            warmup_steps: 1_000_000,
            ..DiagConfig::default()
        };
        let report = run(Arc::new(FrohlichModel::new(alpha)), cfg.clone()).unwrap();
        let e0 = report.e0.unwrap();
        eprintln!("alpha={alpha}: E0={} +/- {}", e0.mean, e0.stderr);
        assert!((e0.mean - target).abs() <= 3.0 * e0.stderr);
        assert!((report.e0_thermo.mean - target).abs() <= 3.0 * report.e0_thermo.stderr);

        let fixed = run(
            Arc::new(FrohlichModel::new(alpha)),
            DiagConfig {
                fixed_tau: true,
                max_tau: 30.0,
                tau_fit: 15.0,
                ..cfg
            },
        )
        .unwrap();
        eprintln!(
            "alpha={alpha}: E_thermo={} +/- {}",
            fixed.e0_thermo.mean, fixed.e0_thermo.stderr
        );
        assert!((fixed.e0_thermo.mean - target).abs() <= 3.0 * fixed.e0_thermo.stderr);
        assert_eq!(report.sign.mean, 1.0);
    }
}

#[test]
#[ignore = "10M-step thermodynamic-estimator curve"]
fn frohlich_alpha_two_energy_curve_flattens() {
    let cfg = DiagConfig {
        mu: -2.3,
        max_tau: 30.0,
        tau_fit: 10.0,
        max_steps: 10_000_000,
        warmup_steps: 1_000_000,
        ..DiagConfig::default()
    };
    let report = run(Arc::new(FrohlichModel::new(2.0)), cfg).unwrap();
    let green = report.green.as_ref().unwrap();
    for (&tau, energy) in green.tau.iter().zip(report.green_energy.as_ref().unwrap()) {
        eprintln!("tau={tau:.3} E={} +/- {}", energy.mean, energy.stderr);
    }
    let e0 = report.e0.unwrap();
    let error = (e0.stderr.powi(2) + report.e0_thermo.stderr.powi(2)).sqrt();
    assert!((e0.mean - report.e0_thermo.mean).abs() <= 3.0 * error);
}

#[test]
fn zero_chains_auto_sizes_to_the_thread_pool() {
    let model: Arc<dyn Model> = Arc::new(FrohlichModel::new(1.0));
    let cfg = DiagConfig {
        chains: 0,
        max_steps: 20_000,
        warmup_steps: 2_000,
        max_tau: 10.0,
        tau_fit: 5.0,
        ..DiagConfig::default()
    };
    let report = run(model, cfg).expect("chains = 0 must auto-size, not fail validation");
    assert!(report.e0.unwrap().mean.is_finite());
    assert!((report.sign.mean - 1.0).abs() < 1.0e-12);
}

#[test]
fn old_config_without_end_time_margin_parses_with_default() {
    let json = serde_json::to_value(DiagConfig::default()).unwrap();
    let mut map = json;
    map.as_object_mut().unwrap().remove("end_time_margin_secs");
    let cfg: DiagConfig = serde_json::from_value(map).unwrap();
    assert_eq!(cfg.end_time_margin_secs, 60.0);
}
