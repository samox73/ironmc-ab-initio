//! CLI: `rmc-minimal [full|bare] [max_steps] [warmup_steps]` (default `full`). `full` samples
//! ⟨x⟩ and ⟨x²⟩ with all three updates; `bare` runs the single-update hot path with no
//! measurement, for engine throughput comparisons. Prints a `steps/sec: <value>` line that
//! `cargo bench-compare` parses.

use std::hint::black_box;
use std::time::{Duration, Instant};

use rmc_core::mc::{
    run_chain, Kernel, Measurement, MetropolisKernel, NoopCallbacks, NullMeasurement, RunCallbacks,
    SimulationCtx, SimulationParams,
};
use rmc_core::random::{ChainId, DefaultRng, SeedSource};
use rmc_core::RmcError;
use rmc_minimal::{build_bare, build_full, minimal_measurement, MinimalState, DEFAULT_BATCH_SIZE};
use rmc_stats::ScalarJackknife;

const DEFAULT_WARMUP_STEPS: u64 = 100_000;
const DEFAULT_MAX_STEPS: u64 = 50_000_000;
const STEPS_PER_CYCLE: u64 = 5;
const SEED: u64 = 0x5eed_5eed_5eed_5eed;
// How often the runner polls the progress callback; the callback itself rate-limits printing.
const PROGRESS_CYCLES_PER_CHECK: u64 = 100_000;

/// Emits `step <done>/<total>` lines on stderr for cargo-bench-compare's
/// `--progress-regex 'step (\d+)/(\d+)'`, rate-limited to ~10 lines/sec so the
/// benchmarked process doesn't pay for its own printing.
struct StderrProgress {
    total: u64,
    last_print: Instant,
}

impl StderrProgress {
    fn new(total: u64) -> Self {
        Self {
            total,
            last_print: Instant::now(),
        }
    }
}

impl RunCallbacks<SimulationCtx> for StderrProgress {
    fn on_checkpoint(&mut self, ctx: &SimulationCtx) {
        if self.last_print.elapsed() >= Duration::from_millis(100) {
            eprintln!("step {}/{}", ctx.steps_done, self.total);
            self.last_print = Instant::now();
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Full,
    Bare,
}

impl Mode {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "full" => Some(Self::Full),
            "bare" => Some(Self::Bare),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Bare => "bare",
        }
    }
}

#[derive(Debug)]
struct RunResult<O> {
    elapsed: Duration,
    steps_done: u64,
    output: O,
}

fn main() -> rmc_core::Result<()> {
    // Silent unless RUST_LOG is set, so benchmark timing stays clean.
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let mode = match args.next() {
        Some(value) => Mode::parse(&value).ok_or_else(|| {
            RmcError::InvalidArgument(format!("mode must be 'full' or 'bare' (got '{value}')"))
        })?,
        None => Mode::Full,
    };
    let max_steps = parse_or_default(args.next(), DEFAULT_MAX_STEPS, "max_steps")?;
    let warmup_steps = parse_or_default(args.next(), DEFAULT_WARMUP_STEPS, "warmup_steps")?;

    println!(
        "mode={} warmup_steps={} max_steps={} steps_per_cycle={}",
        mode.as_str(),
        warmup_steps,
        max_steps,
        STEPS_PER_CYCLE
    );

    match mode {
        Mode::Full => run_full(max_steps, warmup_steps)?,
        Mode::Bare => run_bare(max_steps, warmup_steps)?,
    }

    Ok(())
}

fn parse_or_default<T>(value: Option<String>, default: T, name: &str) -> rmc_core::Result<T>
where
    T: std::str::FromStr,
{
    match value {
        Some(value) => value
            .parse()
            .map_err(|_| RmcError::InvalidArgument(format!("{name} must be a valid value"))),
        None => Ok(default),
    }
}

fn params(max_steps: u64) -> SimulationParams {
    SimulationParams {
        max_steps,
        steps_per_cycle: STEPS_PER_CYCLE,
        cycles_per_check: u64::MAX,
    }
}

fn measured_params(max_steps: u64) -> SimulationParams {
    SimulationParams {
        cycles_per_check: PROGRESS_CYCLES_PER_CHECK,
        ..params(max_steps)
    }
}

fn run_full(max_steps: u64, warmup_steps: u64) -> rmc_core::Result<()> {
    let result = run_once(
        max_steps,
        warmup_steps,
        MetropolisKernel::new(build_full()?),
        minimal_measurement(DEFAULT_BATCH_SIZE)?,
    )?;
    let steps_per_sec = result.steps_done as f64 / result.elapsed.as_secs_f64();
    println!("sample_secs: {:.6}", result.elapsed.as_secs_f64());
    println!("steps/sec: {:.3}", steps_per_sec);
    black_box(result.steps_done);
    black_box(&result.output);
    let (x, x2) = result.output;
    print_observable("x", &x);
    print_observable("x2", &x2);
    Ok(())
}

fn run_bare(max_steps: u64, warmup_steps: u64) -> rmc_core::Result<()> {
    let result = run_once(
        max_steps,
        warmup_steps,
        MetropolisKernel::new(build_bare()?),
        NullMeasurement,
    )?;
    let steps_per_sec = result.steps_done as f64 / result.elapsed.as_secs_f64();
    println!("sample_secs: {:.6}", result.elapsed.as_secs_f64());
    println!("steps/sec: {:.3}", steps_per_sec);
    black_box(result.steps_done);
    black_box(&result.output);
    Ok(())
}

fn run_once<K, M>(
    max_steps: u64,
    warmup_steps: u64,
    mut kernel: K,
    measurement: M,
) -> rmc_core::Result<RunResult<M::Output>>
where
    K: Kernel<MinimalState, DefaultRng>,
    M: Measurement<MinimalState>,
{
    let mut rng = SeedSource::new(SEED).rng_for(ChainId(0));
    let state = MinimalState::default();
    let (state, _, _) = run_chain(
        state,
        &mut rng,
        &mut kernel,
        NullMeasurement,
        params(warmup_steps),
        NoopCallbacks,
    )?;

    let mut progress = StderrProgress::new(max_steps);
    let start = Instant::now();
    let (_state, stats, output) = run_chain(
        state,
        &mut rng,
        &mut kernel,
        measurement,
        measured_params(max_steps),
        &mut progress,
    )?;
    let elapsed = start.elapsed();

    Ok(RunResult {
        elapsed,
        steps_done: stats.steps_done,
        output,
    })
}

fn print_observable(name: &str, value: &ScalarJackknife) {
    let estimate = value.estimate().unwrap_or(f64::NAN);
    let stderr = value.standard_error().unwrap_or(f64::NAN);
    println!("{name}={estimate:.8} stderr={stderr:.8}");
}
