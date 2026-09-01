use std::f64::consts::PI;

use nalgebra::Vector3;
use num_complex::Complex64;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};

/// The only model-specific input required by the diagram engine.
pub trait Model: Send + Sync {
    fn n_branches(&self) -> usize;
    fn electron_energy(&self, k: Vector3<f64>) -> f64;
    fn phonon_energy(&self, q: Vector3<f64>, nu: usize) -> f64;
    fn vertex(&self, k: Vector3<f64>, q: Vector3<f64>, nu: usize) -> Complex64;
    fn propose_phonon(&self, rng: &mut dyn RngCore) -> (Vector3<f64>, usize, f64);
    fn propose_phonon_pdf(&self, q: Vector3<f64>, nu: usize) -> f64;

    /// Minus the logarithm of the per-arc momentum-measure normalization.
    /// Continuum models use d³q/(2π)³; discrete-grid models must return ln(Nq).
    fn log_momentum_measure(&self) -> f64 {
        3.0 * (2.0 * PI).ln()
    }

    /// Modes below this energy must have an identically zero vertex.
    fn phonon_frequency_cutoff(&self) -> f64 {
        0.0
    }
}

/// Fröhlich model in ℏ=m=ω_LO=1 units, provided only as an engine check.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FrohlichModel {
    pub alpha: f64,
}

impl FrohlichModel {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha.is_finite());
        Self { alpha }
    }

    const P0: f64 = std::f64::consts::SQRT_2;
}

impl Model for FrohlichModel {
    fn n_branches(&self) -> usize {
        1
    }

    fn electron_energy(&self, k: Vector3<f64>) -> f64 {
        k.norm_squared() / 2.0
    }

    fn phonon_energy(&self, _q: Vector3<f64>, nu: usize) -> f64 {
        assert_eq!(nu, 0);
        1.0
    }

    fn vertex(&self, _k: Vector3<f64>, q: Vector3<f64>, nu: usize) -> Complex64 {
        assert_eq!(nu, 0);
        let amplitude = (2.0 * 2.0_f64.sqrt() * PI * self.alpha).sqrt() / q.norm();
        Complex64::new(amplitude, 0.0)
    }

    fn propose_phonon(&self, rng: &mut dyn RngCore) -> (Vector3<f64>, usize, f64) {
        let cos_theta = 1.0 - 2.0 * rng.gen::<f64>();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * PI * rng.gen::<f64>();
        let radius = Self::P0 / rng.gen::<f64>().max(f64::MIN_POSITIVE) - Self::P0;
        let q = Vector3::new(
            radius * sin_theta * phi.cos(),
            radius * sin_theta * phi.sin(),
            radius * cos_theta,
        );
        (q, 0, self.propose_phonon_pdf(q, 0))
    }

    fn propose_phonon_pdf(&self, q: Vector3<f64>, nu: usize) -> f64 {
        if nu != 0 {
            return 0.0;
        }
        let radius = q.norm();
        if radius == 0.0 {
            return f64::INFINITY;
        }
        Self::P0 / (4.0 * PI * radius * radius * (radius + Self::P0).powi(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn frohlich_sampler_reports_its_cartesian_density() {
        let model = FrohlichModel::new(1.0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let (q, branch, pdf) = model.propose_phonon(&mut rng);
        assert_eq!(branch, 0);
        assert_eq!(pdf, model.propose_phonon_pdf(q, branch));
        assert!(pdf.is_finite() && pdf > 0.0);
    }
}
