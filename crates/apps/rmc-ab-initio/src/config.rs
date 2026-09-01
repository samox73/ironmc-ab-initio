//! Phase B5: the ab-initio run config — a Fröhlich [`RunConfig`] (engine knobs;
//! its `alpha` is the importance-sampling *envelope* coupling, a calibration
//! knob) plus the material constants and the vertex choice.

use std::path::Path;

use rmc_frohlich::config::RunConfig;

use crate::model::ReducedModel;
use crate::vertex::{radial_from_model, Material, Reweight};

type BoxResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Which vertex the run samples.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VertexSpec {
    /// Analytic Fröhlich `|g|² ∝ α/q²` at coupling `alpha` (the Fröhlich-limit
    /// validation, rungs 0–1). Reweight is the constant `alpha/α_engine`.
    Frohlich { alpha: f64 },
    /// The real LiF tensor, from a cached [`ReducedModel`] JSON (rung 2). Reweight
    /// is the radial `R(|q|)` from [`radial_from_model`].
    Tensor { model_path: String },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct AbInitioConfig {
    /// Fröhlich engine knobs (mu, momentum, tau cutoff, steps, seed, estimators…).
    /// `run.alpha` is the sampling-envelope coupling; the physics is set by `vertex`.
    pub run: RunConfig,
    pub material: Material,
    pub vertex: VertexSpec,
}

impl Default for AbInitioConfig {
    fn default() -> Self {
        // Rung 0: pure Fröhlich at α=1 → E≈-1.016, Z≈0.59.
        Self {
            run: RunConfig::default(),
            material: Material {
                eps_inf: 2.0437,
                eps_zero: 9.0,
                omega_lo_mev: 77.2,
                mass_eff: 0.9,
            },
            vertex: VertexSpec::Frohlich { alpha: 1.0 },
        }
    }
}

impl AbInitioConfig {
    /// The acceptance reweight `R(q)` for this run's vertex (plan B1/B3).
    pub fn build_reweight(&self) -> BoxResult<Reweight> {
        Ok(match &self.vertex {
            VertexSpec::Frohlich { alpha } => Reweight::constant(alpha / self.run.alpha),
            VertexSpec::Tensor { model_path } => {
                let reduced = ReducedModel::load(model_path)?;
                radial_from_model(&reduced, &self.material, self.run.alpha)
            }
        })
    }

    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn load_json(path: impl AsRef<Path>) -> BoxResult<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read config '{}': {e}", path.display()))?;
        Ok(Self::from_json_str(&json)
            .map_err(|e| format!("failed to parse config '{}': {e}", path.display()))?)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> BoxResult<()> {
        std::fs::write(path, self.to_json_string()?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn default_frohlich_reweight_is_unit() {
        let cfg = AbInitioConfig::default();
        // α_target = α_engine = 1 ⇒ R ≡ 1.
        assert_eq!(
            cfg.build_reweight()
                .unwrap()
                .factor(Vector3::new(0.7, 0.0, 0.0)),
            1.0
        );
    }

    #[test]
    fn config_json_round_trips() {
        let cfg = AbInitioConfig::default();
        let s = cfg.to_json_string().unwrap();
        assert_eq!(cfg, AbInitioConfig::from_json_str(&s).unwrap());
    }
}
