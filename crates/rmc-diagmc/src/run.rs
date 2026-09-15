use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use indicatif::{MultiProgress, ProgressBar};
use nalgebra::Vector3;
use rand::Rng;
use rmc_core::mc::{
    default_progress_style, IndicatifProgress, Kernel, MetropolisKernel, Runner, SimulationParams,
    StepOutcome, WithDeadline,
};
use rmc_core::random::{ChainId, DefaultRng, SeedSource};
use rmc_core::{Result, RmcError};
use serde::{Deserialize, Serialize};

use crate::diagram::Diagram;
use crate::measure::{summarize, DiagMeasurement, MeasurementData, Report};
use crate::model::Model;
use crate::updates::{update_set, update_set_fixed_tau};
use crate::weight;

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
    /// Number of independent chains. `0` (or omitting the field in JSON)
    /// auto-sizes to the rayon thread pool — one chain per available thread,
    /// which under slurm follows `RAYON_NUM_THREADS`/the cpuset allocation.
    #[serde(default)]
    pub chains: u64,
    pub max_steps: u64,
    pub warmup_steps: u64,
    pub steps_per_cycle: u64,
    pub n_batches: usize,
    /// Pin the external time to `max_tau` and use only the thermodynamic estimator.
    pub fixed_tau: bool,
    /// Safety margin (seconds) before `SLURM_JOB_END_TIME` at which the
    /// measurement loop stops so the report can be summarized and written.
    /// Inert unless the env var is set.
    #[serde(default = "default_end_time_margin_secs")]
    pub end_time_margin_secs: f64,
}

fn default_end_time_margin_secs() -> f64 {
    60.0
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
            end_time_margin_secs: default_end_time_margin_secs(),
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
            || self.num_bins < 2
            || self.min_order != 0
            || self.max_order == 0
            || self.max_steps == 0
            || self.steps_per_cycle == 0
            || self.n_batches < 2
            || !self.end_time_margin_secs.is_finite()
            || self.end_time_margin_secs < 0.0
        {
            return Err(RmcError::InvalidArgument("invalid DiagConfig".to_string()));
        }
        // The estimator windows select histogram bins by their centers (the last
        // sits at max_tau·(1 − 1/(2·num_bins))), so tau_fit must leave at least
        // the last bin inside the window or every estimate is built from an
        // empty slice.
        if self.tau_fit > self.max_tau * (1.0 - 1.0 / self.num_bins as f64) {
            return Err(RmcError::InvalidArgument(format!(
                "tau_fit = {} leaves no histogram bin in the fit window; it must \
                 be at most max_tau·(1 − 1/num_bins) = {}",
                self.tau_fit,
                self.max_tau * (1.0 - 1.0 / self.num_bins as f64)
            )));
        }
        Ok(())
    }
}

/// Deadline derived from `SLURM_JOB_END_TIME` (epoch seconds, exported by
/// slurm ≥ 22.05), minus the safety margin. Read once at startup; a mid-run
/// `scontrol update TimeLimit` is not tracked. No env var → `None`, feature
/// off. Public: the flat/GPU batched drivers arm the same deadline.
pub fn slurm_deadline(margin_secs: f64) -> Result<Option<Instant>> {
    let Ok(raw) = std::env::var("SLURM_JOB_END_TIME") else {
        return Ok(None);
    };
    let end: u64 = raw.parse().map_err(|_| {
        RmcError::InvalidArgument(format!(
            "SLURM_JOB_END_TIME is not epoch seconds: {raw:?}"
        ))
    })?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before 1970")
        .as_secs_f64();
    let remaining = end as f64 - now - margin_secs;
    if remaining <= 0.0 {
        return Err(RmcError::InvalidState(format!(
            "SLURM job ends within the {margin_secs} s end-time margin; refusing to start"
        )));
    }
    log::info!("SLURM end-time deadline: stopping in {remaining:.0} s (margin {margin_secs} s)");
    Ok(Some(Instant::now() + Duration::from_secs_f64(remaining)))
}

/// One leg of a checkpointed / multi-rank slotmap run — the CPU-fast engine.
/// `data` is this leg's raw measurements (for a `RankDump`); `chains` is every
/// chain's final `(Diagram, rng)`, which a checkpoint saves so the next leg
/// continues bit-identically.
pub struct SlotmapLeg {
    pub report: Report,
    pub data: MeasurementData,
    pub steps_done: u64,
    pub chains: Vec<(Diagram, DefaultRng)>,
    /// This leg ran its full `max_steps` per chain (i.e. the deadline did not
    /// truncate it).
    pub complete: bool,
}

pub fn run(model: Arc<dyn Model>, cfg: DiagConfig) -> Result<Report> {
    Ok(run_leg(model, cfg, 0, None)?.report)
}

/// The slotmap run parameterized for multi-node (chain-id `chain_offset`) and
/// checkpoint resume (`resume` = each chain's saved `(Diagram, rng)`; when
/// `Some`, warmup is skipped and the chains continue from where they stopped).
/// Rank `r` passes `chain_offset = r * cfg.chains`; the resulting `N * chains`
/// streams are all distinct, so merging the ranks equals one big run.
pub fn run_leg(
    model: Arc<dyn Model>,
    cfg: DiagConfig,
    chain_offset: u64,
    resume: Option<Vec<(Diagram, DefaultRng)>>,
) -> Result<SlotmapLeg> {
    cfg.validate()?;
    let deadline = slurm_deadline(cfg.end_time_margin_secs)?;
    if model.n_branches() == 0 {
        return Err(RmcError::InvalidArgument(
            "model must contain at least one phonon branch".to_string(),
        ));
    }

    let chains = if cfg.chains == 0 {
        // `chain_offset = rank * chains` would collide across ranks if chains
        // auto-sized per node, and a resumed run must match its checkpoint's
        // chain count — both require an explicit, node-count-independent pin.
        if chain_offset != 0 || resume.is_some() {
            return Err(RmcError::InvalidArgument(
                "multi-rank or resumed runs must pin `chains` explicitly \
                 (chains = 0 auto-sizing would break chain-id offsets)"
                    .to_string(),
            ));
        }
        let auto = rayon::current_num_threads() as u64;
        log::info!("chains = 0 → auto-sized to {auto} (one chain per rayon thread)");
        auto
    } else {
        cfg.chains
    };
    if let Some(resume) = &resume {
        if resume.len() != chains as usize {
            return Err(RmcError::InvalidArgument(format!(
                "checkpoint has {} chains, config wants {chains}",
                resume.len()
            )));
        }
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
        weight::sync_cache(&mut diagram, build_model.as_ref());
        let updates = if build_cfg.fixed_tau {
            update_set_fixed_tau(build_model.clone())
        } else {
            update_set(build_model.clone())
        };
        let kernel = DriftCheckedKernel {
            inner: MetropolisKernel::new(updates.expect("validated update set")),
            model: build_model.clone(),
            steps: 0,
            interval: build_cfg.steps_per_cycle.saturating_mul(4096),
        };
        let measurement = DiagMeasurement::new(
            build_model.clone(),
            build_cfg.num_bins,
            build_cfg.n_batches,
            expected_samples,
            build_cfg.max_tau,
        );
        (diagram, kernel, measurement)
    })
    .chains(chains)
    .chain_offset(chain_offset)
    .warmup(SimulationParams {
        max_steps: cfg.warmup_steps,
        steps_per_cycle: cfg.steps_per_cycle,
        cycles_per_check: 0,
    });

    // Per-chain progress bars (hidden automatically when stderr is not a tty).
    let multi = MultiProgress::new();
    let progress = |total_steps: u64, label: String, finish: String| {
        let bar = multi.add(ProgressBar::new(total_steps));
        bar.set_style(default_progress_style());
        bar.set_prefix(label);
        IndicatifProgress::new(bar).with_finish_message(finish)
    };
    let runner = runner
        .warmup_callbacks(|chain: ChainId| {
            progress(
                cfg.warmup_steps,
                format!("warmup {}", chain.0),
                format!("warmup {} done", chain.0),
            )
        })
        .callbacks(|chain: ChainId| WithDeadline {
            inner: progress(
                cfg.max_steps,
                format!("chain {}", chain.0),
                format!("chain {} done", chain.0),
            ),
            deadline,
        });

    let output = runner.run_resumable(
        SimulationParams {
            max_steps: cfg.max_steps,
            steps_per_cycle: cfg.steps_per_cycle,
            // 4096 cycles between deadline checks matches the drift-check cadence;
            // 0 keeps the loop check-free when no deadline is set.
            cycles_per_check: if deadline.is_some() { 4096 } else { 0 },
        },
        resume,
    )?;
    let total_steps = chains * cfg.max_steps;
    let complete = output.stats.steps_done >= total_steps;
    if !complete {
        log::warn!(
            "deadline stop: {}/{} measurement steps completed ({:.1}%); \
             error bars are honest but larger, and the sample count is \
             wall-clock-dependent (not reproducible from the seed alone)",
            output.stats.steps_done,
            total_steps,
            100.0 * output.stats.steps_done as f64 / total_steps as f64
        );
    }
    let data = output.output;
    let mut report = summarize(
        data.clone(),
        model.electron_energy(cfg.p_ext),
        cfg.mu,
        cfg.max_tau,
        cfg.tau_fit,
        cfg.fixed_tau,
    )?;
    report.steps_done = output.stats.steps_done;
    let chain_states: Vec<(Diagram, DefaultRng)> =
        output.states.into_iter().zip(output.rngs).collect();
    Ok(SlotmapLeg {
        report,
        data,
        steps_done: output.stats.steps_done,
        chains: chain_states,
        complete,
    })
}

struct DriftCheckedKernel<S> {
    inner: MetropolisKernel<S>,
    model: Arc<dyn Model>,
    steps: u64,
    interval: u64,
}

impl<R: Rng, S> Kernel<Diagram, R> for DriftCheckedKernel<S>
where
    MetropolisKernel<S>: Kernel<Diagram, R>,
{
    fn prepare(&mut self, diagram: &mut Diagram) -> Result<()> {
        self.inner.prepare(diagram)
    }

    fn step(&mut self, diagram: &mut Diagram, rng: &mut R) -> Result<StepOutcome> {
        let outcome = self.inner.step(diagram, rng)?;
        self.steps += 1;
        if self.steps % self.interval == 0 {
            // Structural audit (link symmetry, τ ordering, momenta re-derived
            // from the arc q's). This is the only momentum check that survives
            // release builds, where the per-mutation debug_asserts vanish.
            assert!(
                diagram.check_consistency(),
                "diagram bookkeeping (links/ordering/momenta) is inconsistent"
            );
            let cached_log = diagram.cached_log_modulus();
            let cached_phase = diagram.cached_phase();
            let (log_modulus, phase) = weight::log_modulus_and_phase(diagram, self.model.as_ref());
            // Relative tolerance: the cache accumulates one rounding error per
            // accepted move between resyncs, proportional to |log W|.
            let tolerance = 1.0e-8 * (1.0 + log_modulus.abs());
            assert!(
                (cached_log - log_modulus).abs() < tolerance,
                "cached log-modulus drifted: {cached_log} vs {log_modulus}"
            );
            assert!((cached_phase - phase).norm() < 1.0e-8);
            diagram.set_cached_weight(log_modulus, phase);
        }
        Ok(outcome)
    }

    fn reset_stats(&mut self) {
        self.inner.reset_stats();
    }
}
