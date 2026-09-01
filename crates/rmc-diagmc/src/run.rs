use std::sync::Arc;

use nalgebra::Vector3;
use rmc_core::mc::{MetropolisKernel, Runner, SimulationParams};
use rmc_core::random::SeedSource;
use rmc_core::{Result, RmcError};
use serde::{Deserialize, Serialize};

use crate::diagram::Diagram;
use crate::measure::{summarize, DiagMeasurement, Report};
use crate::model::Model;
use crate::updates::{update_set, update_set_fixed_tau};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DiagConfig {
    /// Fictitious chemical-potential shift. It cancels from fixed-τ estimates.
    pub mu: f64,
    pub p_ext: Vector3<f64>,
    pub max_tau: f64,
    pub num_bins: usize,
    pub min_order: usize,
    pub max_order: usize,
    pub tau_fit: f64,
    pub seed: u64,
    pub chains: u64,
    pub max_steps: u64,
    pub warmup_steps: u64,
    pub steps_per_cycle: u64,
    pub n_batches: usize,
    /// Pin the external time to `max_tau` and use only the thermodynamic estimator.
    pub fixed_tau: bool,
}

impl Default for DiagConfig {
    fn default() -> Self {
        Self {
            mu: -1.2,
            p_ext: Vector3::zeros(),
            max_tau: 20.0,
            num_bins: 80,
            min_order: 0,
            max_order: 100,
            tau_fit: 10.0,
            seed: 1,
            chains: 4,
            max_steps: 1_000_000,
            warmup_steps: 100_000,
            steps_per_cycle: 10,
            n_batches: 20,
            fixed_tau: false,
        }
    }
}

impl DiagConfig {
    fn validate(&self) -> Result<()> {
        if !self.mu.is_finite()
            || !self.max_tau.is_finite()
            || self.max_tau <= 0.0
            || !self.tau_fit.is_finite()
            || self.tau_fit < 0.0
            || self.tau_fit >= self.max_tau
            || self.num_bins < 2
            || self.min_order != 0
            || self.max_order == 0
            || self.chains == 0
            || self.max_steps == 0
            || self.steps_per_cycle == 0
            || self.n_batches < 2
        {
            return Err(RmcError::InvalidArgument("invalid DiagConfig".to_string()));
        }
        Ok(())
    }
}

pub fn run(model: Arc<dyn Model>, cfg: DiagConfig) -> Result<Report> {
    cfg.validate()?;
    if model.n_branches() == 0 {
        return Err(RmcError::InvalidArgument(
            "model must contain at least one phonon branch".to_string(),
        ));
    }

    let expected_samples = cfg.max_steps.div_ceil(cfg.steps_per_cycle) as usize;
    let build_cfg = cfg.clone();
    let build_model = model.clone();
    let runner = Runner::new(SeedSource::new(cfg.seed), move |_chain| {
        let mut diagram = Diagram::new(
            build_cfg.mu,
            build_cfg.p_ext,
            build_cfg.max_tau,
            build_cfg.min_order,
            build_cfg.max_order,
        );
        if build_cfg.fixed_tau {
            let tail = diagram.tail;
            diagram.set_vertex_tau(tail, build_cfg.max_tau);
        }
        let updates = if build_cfg.fixed_tau {
            update_set_fixed_tau(build_model.clone())
        } else {
            update_set(build_model.clone())
        };
        let kernel = MetropolisKernel::new(updates.expect("validated update set"));
        let measurement = DiagMeasurement::new(
            build_model.clone(),
            build_cfg.num_bins,
            build_cfg.n_batches,
            expected_samples,
            build_cfg.max_tau,
        );
        (diagram, kernel, measurement)
    })
    .chains(cfg.chains)
    .warmup(SimulationParams {
        max_steps: cfg.warmup_steps,
        steps_per_cycle: cfg.steps_per_cycle,
        cycles_per_check: 0,
    });

    let output = runner.run(SimulationParams {
        max_steps: cfg.max_steps,
        steps_per_cycle: cfg.steps_per_cycle,
        cycles_per_check: 0,
    })?;
    summarize(
        output.output,
        model.as_ref(),
        cfg.mu,
        cfg.p_ext,
        cfg.max_tau,
        cfg.tau_fit,
        cfg.fixed_tau,
    )
}
