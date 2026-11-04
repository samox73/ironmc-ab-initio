//! Phase B driver: run the reused Fröhlich engine with the material reweight
//! installed, then report in physical units via the B4 bridge.
//!
//! Runs `cfg.run.chains` independent chains via [`Runner`] (merged statistics —
//! important because the strong-coupling estimator is heavy-tailed, so several
//! chains give honest error bars). Everything but [`crate::updates::update_set`]
//! is reused from `rmc-frohlich`.

use std::path::Path;
use std::time::Instant;

use rmc_core::mc::{MetropolisKernel, Runner, UpdateSet, WeightedUpdateSet};
use rmc_core::random::{ChainId, SeedSource};
use rmc_frohlich::Diagram;
use rmc_frohlich::app::{
    RunOutput, ValidationSummary, build_diagram, build_measurement, summarize_output, write_results,
};
use rmc_frohlich::measurement::PolaronMeasurement;
use rmc_frohlich::update_stats::{self, UpdateStatEntry};

use crate::config::AbInitioConfig;
use crate::updates::{AbInitioUpdate, update_set};
use crate::vertex::Reweight;

type BoxResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub type AbInitioKernel = MetropolisKernel<WeightedUpdateSet<AbInitioUpdate>>;

/// Physical-unit summary on top of the dimensionless Fröhlich [`ValidationSummary`].
#[derive(Clone, Debug)]
pub struct AbInitioReport {
    pub summary: ValidationSummary,
    pub alpha_lif: f64,
    /// Ground-state energy in meV (`E_Fröhlich · ħω_LO`).
    pub energy_mev: f64,
    pub energy_stderr_mev: f64,
}

impl AbInitioReport {
    pub fn text(&self) -> String {
        format!(
            "{}\n---- ab-initio (physical units) ----\nalpha_LiF: {:.4}\nE0: {:.3} +/- {:.3} meV",
            self.summary.text(),
            self.alpha_lif,
            self.energy_mev,
            self.energy_stderr_mev,
        )
    }
}

/// Per-chain builder: the reused Fröhlich diagram + measurement, and our kernel
/// with the material reweight installed.
fn build_chain(
    cfg: AbInitioConfig,
    reweight: Reweight,
) -> impl Fn(ChainId) -> (Diagram, AbInitioKernel, PolaronMeasurement) {
    move |_chain| {
        let diagram = build_diagram(&cfg.run);
        let kernel = MetropolisKernel::new(
            update_set(reweight.clone()).expect("ab-initio update set must be valid"),
        );
        let measurement = build_measurement(&cfg.run, &diagram);
        (diagram, kernel, measurement)
    }
}

/// Run `cfg.run.chains` chains (with warmup) and produce the report plus the raw
/// merged [`RunOutput`].
pub fn run(cfg: &AbInitioConfig) -> BoxResult<(AbInitioReport, RunOutput)> {
    let reweight = cfg.build_reweight()?;
    let runner = Runner::new(
        SeedSource::new(cfg.run.seed),
        build_chain(cfg.clone(), reweight),
    )
    .chains(cfg.run.chains)
    .warmup(cfg.run.warmup_params());

    let start = Instant::now();
    let report = runner.run(cfg.run.simulation_params())?;
    let wall_secs = start.elapsed().as_secs_f64();

    let output = RunOutput {
        stats: report.stats,
        measurement: report.output,
        final_state: (cfg.run.chains == 1).then(|| report.states.into_iter().next().unwrap()),
        update_stats: update_stats::merge(report.kernels.iter().map(collect).collect()),
        wall_secs,
    };
    let summary = summarize_output(&cfg.run, &output);
    let report = AbInitioReport {
        alpha_lif: cfg.material.frohlich_alpha(),
        energy_mev: cfg.material.energy_to_mev(summary.energy.mean),
        energy_stderr_mev: cfg.material.energy_to_mev(summary.energy.stderr),
        summary,
    };
    Ok((report, output))
}

/// Run and also write the JSON result bundle into `dir`.
pub fn run_and_write(cfg: &AbInitioConfig, dir: impl AsRef<Path>) -> BoxResult<AbInitioReport> {
    let (report, output) = run(cfg)?;
    write_results(&cfg.run, &output, dir)?;
    Ok(report)
}

/// Per-update proposal/acceptance table (local twin of `rmc_frohlich::update_stats::collect`,
/// which is typed to the Fröhlich kernel).
fn collect(kernel: &AbInitioKernel) -> Vec<UpdateStatEntry> {
    kernel
        .updates()
        .entries()
        .iter()
        .zip(kernel.updates().stats())
        .map(|(entry, s)| UpdateStatEntry {
            name: entry.update().name().to_string(),
            weight: entry.weight(),
            proposed: s.nprops,
            impossible: s.nimps,
            accepted: s.naccs,
            acc_ratio: if s.nprops > 0 { s.naccs as f64 / s.nprops as f64 } else { 0.0 },
        })
        .collect()
}
