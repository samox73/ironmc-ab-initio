use rayon::prelude::*;
use rayon::ThreadPool;

use crate::random::{ChainId, DefaultRng, SeedSource};
use crate::{Merge, Result, RmcError};

use super::run::{run_chain, NoopCallbacks, NullMeasurement, SimulationParams, SimulationStats};
use super::traits::{Kernel, Measurement, RunCallbacks};

/// Everything a finished run produces. Ignore the fields you don't need.
pub struct RunReport<State, K, O> {
    pub stats: SimulationStats,
    pub output: O,
    pub kernels: Vec<K>,
    pub states: Vec<State>,
    /// Final per-chain RNG state (same order as `states`), for checkpointing a
    /// run and resuming it bit-identically via [`Runner::run_resumable`].
    pub rngs: Vec<DefaultRng>,
}

type NoCallbacks = fn(ChainId) -> NoopCallbacks;

/// Builder for orchestrated Monte Carlo runs.
pub struct Runner<'p, B, CB = NoCallbacks, WCB = NoCallbacks> {
    chains: u64,
    offset: u64,
    seed: SeedSource,
    warmup: Option<SimulationParams>,
    pool: Option<&'p ThreadPool>,
    build: B,
    callbacks: CB,
    warmup_callbacks: WCB,
}

impl<B> Runner<'_, B> {
    pub fn new(seed: SeedSource, build: B) -> Self {
        Self {
            chains: 1,
            offset: 0,
            seed,
            warmup: None,
            pool: None,
            build,
            callbacks: (|_| NoopCallbacks) as NoCallbacks,
            warmup_callbacks: (|_| NoopCallbacks) as NoCallbacks,
        }
    }
}

impl<'p, B, CB, WCB> Runner<'p, B, CB, WCB> {
    pub fn chains(mut self, chains: u64) -> Self {
        self.chains = chains;
        self
    }

    pub fn warmup(mut self, params: SimulationParams) -> Self {
        self.warmup = Some(params);
        self
    }

    pub fn pool(mut self, pool: &'p ThreadPool) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Global chain-id offset (default 0). Rank `r` of a multi-node run uses
    /// `offset = r * chains`, so all `N * chains` streams are distinct — exactly
    /// as if a single process had run them all.
    pub fn chain_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    pub fn callbacks<CB2>(self, callbacks: CB2) -> Runner<'p, B, CB2, WCB> {
        Runner {
            chains: self.chains,
            offset: self.offset,
            seed: self.seed,
            warmup: self.warmup,
            pool: self.pool,
            build: self.build,
            callbacks,
            warmup_callbacks: self.warmup_callbacks,
        }
    }

    pub fn warmup_callbacks<WCB2>(self, warmup_callbacks: WCB2) -> Runner<'p, B, CB, WCB2> {
        Runner {
            chains: self.chains,
            offset: self.offset,
            seed: self.seed,
            warmup: self.warmup,
            pool: self.pool,
            build: self.build,
            callbacks: self.callbacks,
            warmup_callbacks,
        }
    }

    pub fn run<State, K, M, C, WC>(
        self,
        params: SimulationParams,
    ) -> Result<RunReport<State, K, M::Output>>
    where
        State: Send,
        K: Kernel<State, DefaultRng> + Send,
        M: Measurement<State> + Send,
        M::Output: Merge + Send,
        B: Fn(ChainId) -> (State, K, M) + Sync,
        C: RunCallbacks<super::run::SimulationCtx> + Send,
        WC: RunCallbacks<super::run::SimulationCtx> + Send,
        CB: Fn(ChainId) -> C + Sync,
        WCB: Fn(ChainId) -> WC + Sync,
    {
        self.run_resumable(params, None)
    }

    /// Like [`run`](Self::run), but each chain can resume from a saved
    /// `(state, rng)` (skipping the build + warmup) instead of starting fresh.
    /// `resume`, when given, is indexed by local chain (`0..chains`) and its
    /// length must equal `chains`. Every chain's final `(state, rng)` is returned
    /// in [`RunReport`] (`states`/`rngs`, same order), which is exactly what a
    /// checkpoint needs to continue the run bit-identically.
    pub fn run_resumable<State, K, M, C, WC>(
        self,
        params: SimulationParams,
        resume: Option<Vec<(State, DefaultRng)>>,
    ) -> Result<RunReport<State, K, M::Output>>
    where
        State: Send,
        K: Kernel<State, DefaultRng> + Send,
        M: Measurement<State> + Send,
        M::Output: Merge + Send,
        B: Fn(ChainId) -> (State, K, M) + Sync,
        C: RunCallbacks<super::run::SimulationCtx> + Send,
        WC: RunCallbacks<super::run::SimulationCtx> + Send,
        CB: Fn(ChainId) -> C + Sync,
        WCB: Fn(ChainId) -> WC + Sync,
    {
        if self.chains == 0 {
            return Err(RmcError::InvalidArgument("chains must be > 0".to_string()));
        }
        params.validate()?;
        if let Some(warmup) = &self.warmup {
            warmup.validate()?;
        }
        if let Some(resume) = &resume {
            if resume.len() != self.chains as usize {
                return Err(RmcError::InvalidArgument(format!(
                    "resume has {} chains, runner is configured for {}",
                    resume.len(),
                    self.chains
                )));
            }
        }

        // Run one chain and return its final (state, rng) for checkpointing.
        // Fresh: build + per-chain-seeded rng + warmup. Resumed: use the saved
        // (state, rng), skip warmup (build is still called for a fresh
        // kernel/measurement; its state is discarded).
        let run_one = |chain_id: ChainId,
                       resume_state: Option<(State, DefaultRng)>|
         -> Result<(SimulationStats, M::Output, K, State, DefaultRng)> {
            let (built_state, mut kernel, measurement) = (self.build)(chain_id);
            let (mut state, mut rng, do_warmup) = match resume_state {
                Some((state, rng)) => (state, rng, false),
                None => (built_state, self.seed.rng_for(chain_id), true),
            };
            if do_warmup {
                if let Some(warmup) = self.warmup {
                    if warmup.max_steps > 0 {
                        let callbacks = (self.warmup_callbacks)(chain_id);
                        let (warm_state, _stats, ()) = run_chain(
                            state,
                            &mut rng,
                            &mut kernel,
                            NullMeasurement,
                            warmup,
                            callbacks,
                        )?;
                        state = warm_state;
                        kernel.reset_stats();
                    }
                }
            }
            let callbacks = (self.callbacks)(chain_id);
            let (state, stats, output) =
                run_chain(state, &mut rng, &mut kernel, measurement, params, callbacks)?;
            Ok((stats, output, kernel, state, rng))
        };

        let offset = self.offset;
        let work = || match resume {
            Some(resume) => resume
                .into_par_iter()
                .enumerate()
                .map(|(local, saved)| run_one(ChainId(offset + local as u64), Some(saved)))
                .collect::<Vec<_>>(),
            None => (0..self.chains)
                .into_par_iter()
                .map(|chain| run_one(ChainId(offset + chain), None))
                .collect::<Vec<_>>(),
        };
        let partials = match self.pool {
            Some(pool) => pool.install(work),
            None => work(),
        };

        let mut merged: Option<(SimulationStats, M::Output)> = None;
        let mut kernels = Vec::with_capacity(partials.len());
        let mut states = Vec::with_capacity(partials.len());
        let mut rngs = Vec::with_capacity(partials.len());
        for partial in partials {
            let (stats, output, kernel, state, rng) = partial?;
            kernels.push(kernel);
            states.push(state);
            rngs.push(rng);
            merged = Some(match merged {
                Some((acc_stats, acc_output)) => (acc_stats.merge(stats), acc_output.merge(output)),
                None => (stats, output),
            });
        }
        let (stats, output) =
            merged.ok_or_else(|| RmcError::InvalidState("run produced no chains".to_string()))?;
        Ok(RunReport {
            stats,
            output,
            kernels,
            states,
            rngs,
        })
    }
}
