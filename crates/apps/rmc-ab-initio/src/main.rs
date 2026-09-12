use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rmc_ab_initio::config::AbInitioConfig;
use rmc_ab_initio::eph_svd::{EphSvd, DEFAULT_TOL};
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::model::{regular_grid, ReducedModel};
use rmc_ab_initio::model_abinitio::AbInitioModel;
use rmc_ab_initio::reconstruct::GammaModel;
use rmc_ab_initio::run::run_and_write;
use rmc_ab_initio::vertex_table::{
    TablePrecision, VertexTable, DEFAULT_MAX_BYTES, DEFAULT_PHFREQ_CUTOFF_EV,
};
use rmc_diagmc::{DiagConfig, Model};

type BoxResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Build the reduced `|g(Γ,q,LO)|²` artifact from an EPW dataset dir (holding
/// `epwdata.fmt`, `crystal.fmt`, `wigner.fmt`, `out/LiF.epmatwp`) on a regular
/// `n×n×n` q-grid. Needs the ~146 MB `epmatwp` present on disk.
fn build_reduced(dataset: &Path, n: usize) -> Result<ReducedModel, Box<dyn std::error::Error>> {
    let e = EpwData::read(dataset.join("epwdata.fmt"))?;
    let c = Crystal::read(dataset.join("crystal.fmt"))?;
    let w = Wigner::read(dataset.join("wigner.fmt"))?;
    let wp = EpmatWp::read(
        dataset.join("out/LiF.epmatwp"),
        e.nbnd,
        e.nrr_k,
        e.nmodes,
        e.nrr_g,
    )?;
    if !wp.data[0].is_finite() {
        return Err("epmatwp is all-NaN; regenerate the EPW e-ph tensor".into());
    }
    let gamma = GammaModel::build(&e, &c, &w, &wp);
    Ok(ReducedModel::build(&gamma, &e, &c, &w, &regular_grid(n)))
}

fn epmatwp_path(dataset: &Path) -> BoxResult<PathBuf> {
    let lif = dataset.join("out/LiF.epmatwp");
    if lif.exists() {
        return Ok(lif);
    }
    let mut matches = std::fs::read_dir(dataset.join("out"))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "epmatwp")
        });
    let path = matches
        .next()
        .ok_or("no .epmatwp file found in dataset/out")?;
    if matches.next().is_some() {
        return Err("multiple .epmatwp files found in dataset/out".into());
    }
    Ok(path)
}

fn load_vertex_inputs(dataset: &Path) -> BoxResult<(EpwData, Crystal, Wigner, EpmatWp)> {
    let epw = EpwData::read(dataset.join("epwdata.fmt"))?;
    let crystal = Crystal::read(dataset.join("crystal.fmt"))?;
    let wigner = Wigner::read(dataset.join("wigner.fmt"))?;
    let epmat = EpmatWp::read(
        epmatwp_path(dataset)?,
        epw.nbnd,
        epw.nrr_k,
        epw.nmodes,
        epw.nrr_g,
    )?;
    if !epmat.data.first().is_some_and(|value| value.is_finite()) {
        return Err("epmatwp is empty or non-finite; regenerate the EPW e-ph tensor".into());
    }
    Ok((epw, crystal, wigner, epmat))
}

fn options(args: impl Iterator<Item = String>) -> BoxResult<BTreeMap<String, String>> {
    let mut args = args;
    let mut options = BTreeMap::new();
    while let Some(name) = args.next() {
        if !name.starts_with("--") {
            return Err(format!("expected an option, got '{name}'").into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for '{name}'"))?;
        options.insert(name, value);
    }
    Ok(options)
}

fn take<T: std::str::FromStr>(
    options: &mut BTreeMap<String, String>,
    name: &str,
    default: Option<T>,
) -> BoxResult<T>
where
    T::Err: std::fmt::Display,
{
    match options.remove(name) {
        Some(value) => value
            .parse()
            .map_err(|error| format!("invalid {name} value '{value}': {error}").into()),
        None => default.ok_or_else(|| format!("missing required option {name}").into()),
    }
}

fn gib_to_bytes(gib: f64) -> BoxResult<u64> {
    if !gib.is_finite() || gib <= 0.0 || gib > u64::MAX as f64 / 2_f64.powi(30) {
        return Err("max GiB must be finite, positive, and representable".into());
    }
    Ok((gib * 2_f64.powi(30)) as u64)
}

fn env_table_max_bytes() -> BoxResult<Option<u64>> {
    std::env::var("RMC_TABLE_MAX_GIB")
        .ok()
        .map(|value| {
            let gib = value
                .parse()
                .map_err(|error| format!("invalid RMC_TABLE_MAX_GIB '{value}': {error}"))?;
            gib_to_bytes(gib)
        })
        .transpose()
}

fn table_command(args: impl Iterator<Item = String>) -> BoxResult<()> {
    let mut options = options(args)?;
    let dataset = PathBuf::from(
        options
            .remove("--dataset")
            .unwrap_or_else(|| "datasets/LiF".to_string()),
    );
    let grid_n: usize = take(&mut options, "--grid", None)?;
    let total_rank: usize = take(&mut options, "--total-rank", None)?;
    let precision: TablePrecision = take(&mut options, "--precision", Some(TablePrecision::F32))?;
    if grid_n == 0 || total_rank == 0 {
        return Err("grid and total rank must be positive".into());
    }
    let max_bytes = match options.remove("--max-gib") {
        Some(value) => gib_to_bytes(value.parse()?)?,
        None => env_table_max_bytes()?.unwrap_or(DEFAULT_MAX_BYTES),
    };
    let out = options.remove("--out").map_or_else(
        || dataset.join(VertexTable::cache_filename(grid_n, total_rank, precision)),
        PathBuf::from,
    );
    if let Some(unknown) = options.keys().next() {
        return Err(format!("unknown option {unknown}").into());
    }

    let (epw, crystal, wigner, epmat) = load_vertex_inputs(&dataset)?;
    let started = Instant::now();
    let svd = EphSvd::compress_global(&epmat, DEFAULT_TOL, total_rank);
    println!("compression time: {:.3} s", started.elapsed().as_secs_f64());
    let bytes = VertexTable::estimated_bytes(grid_n, svd.rank(), epw.nbnd, epw.nmodes, precision);
    println!(
        "table estimate: {:.2} GiB (grid={grid_n}, components={}, precision={precision:?})",
        bytes as f64 / 2_f64.powi(30),
        svd.rank()
    );
    println!("output: {}", out.display());
    let started = Instant::now();
    let table = VertexTable::build(
        &epw,
        &crystal,
        &wigner,
        &svd,
        grid_n,
        DEFAULT_PHFREQ_CUTOFF_EV,
        precision,
        max_bytes,
    )?;
    println!("build time: {:.3} s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    table.save(&out)?;
    println!(
        "wrote {} in {:.3} s",
        out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn table_run_command(args: impl Iterator<Item = String>) -> BoxResult<()> {
    let mut options = options(args)?;
    let table_path: PathBuf = take(&mut options, "--table", None)?;
    let config = match options.remove("--config") {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => DiagConfig::default(),
    };
    if let Some(unknown) = options.keys().next() {
        return Err(format!("unknown option {unknown}").into());
    }
    let table = Arc::new(VertexTable::load(table_path)?);
    let started = Instant::now();
    let report = rmc_diagmc::run(Arc::new(AbInitioModel::new(table)), config)?;
    let wall_secs = started.elapsed().as_secs_f64();
    // steps_done is the honest count under a deadline stop; wall includes
    // warmup and summarize (percent-level for production shapes).
    log::info!(
        "DMC complete: {wall_secs:.3} s, {:.3e} steps/sec across all chains",
        report.steps_done as f64 / wall_secs
    );
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn load_config(path: impl AsRef<Path>) -> BoxResult<AbInitioConfig> {
    let mut config = AbInitioConfig::load_json(path)?;
    if let Some(max_bytes) = env_table_max_bytes()? {
        config.table_max_bytes = max_bytes;
    }
    Ok(config)
}

/// Phase A3 batched flat-engine run (plan `gpu-abinitio`): one rank of
/// lockstep chains, RankDump output for the offline `merge`.
fn run_flat_command(args: impl Iterator<Item = String>) -> BoxResult<()> {
    use rmc_ab_initio::dump::RankDump;
    use rmc_ab_initio::flat::batched::{
        run_batched, DEFAULT_GROUP_SIZE, DEFAULT_MAX_ORDER_GPU,
    };

    let mut options = options(args)?;
    let table_path: PathBuf = take(&mut options, "--table", None)?;
    let config: DiagConfig = match options.remove("--config") {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => DiagConfig::default(),
    };
    let rank: u64 = take(&mut options, "--rank", Some(0))?;
    let group: usize = take(&mut options, "--group", Some(DEFAULT_GROUP_SIZE))?;
    let max_order_gpu: usize =
        take(&mut options, "--max-order-gpu", Some(DEFAULT_MAX_ORDER_GPU))?;
    let out_dir: PathBuf = take(&mut options, "--out-dir", Some(PathBuf::from(".")))?;
    if let Some(unknown) = options.keys().next() {
        return Err(format!("unknown option {unknown}").into());
    }

    let table = Arc::new(VertexTable::load(table_path)?);
    let model = Arc::new(AbInitioModel::new(table));
    let chain_offset = rank * config.chains;
    let started = Instant::now();
    let output = run_batched(&model, &config, chain_offset, group, max_order_gpu)?;
    let wall_secs = started.elapsed().as_secs_f64();

    std::fs::create_dir_all(&out_dir)?;
    let dump_path = out_dir.join(RankDump::filename(rank, 0));
    RankDump {
        electron_energy_pext: model.electron_energy(config.p_ext),
        chains: config.chains,
        steps_done: output.steps_done,
        data: output.data,
        config,
        rank,
        leg: 0,
    }
    .save(&dump_path)?;
    log::info!("wrote {}", dump_path.display());

    // Timers in the summary itself, not only logs; steps/s is step-loop-only.
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "report": output.report,
            "update_stats": output.update_stats,
            "step_secs": output.step_secs,
            "replay_secs": output.replay_secs,
            "wall_secs": wall_secs,
            "steps_per_sec_step_loop": output.steps_done as f64 / output.step_secs,
        }))?
    );
    Ok(())
}

/// Merge rank dumps into one report: `merge [--out report.json] <dumps...>`.
fn merge_command(args: impl Iterator<Item = String>) -> BoxResult<()> {
    use rmc_ab_initio::dump::{merge_dumps, RankDump};

    let mut out: Option<PathBuf> = None;
    let mut paths = Vec::new();
    let mut args = args;
    while let Some(arg) = args.next() {
        if arg == "--out" {
            out = Some(PathBuf::from(args.next().ok_or("missing value for --out")?));
        } else {
            paths.push(PathBuf::from(arg));
        }
    }
    if paths.is_empty() {
        return Err("usage: merge [--out report.json] <batches-rank*.bin...>".into());
    }
    let dumps = paths
        .iter()
        .map(RankDump::load)
        .collect::<Result<Vec<_>, _>>()?;
    let report = merge_dumps(dumps)?;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out) = out {
        std::fs::write(&out, &json)?;
        log::info!("wrote {}", out.display());
    }
    println!("{json}");
    Ok(())
}

/// Phase A4/A5 device run: run-flat semantics on the selected GPU runtime,
/// same RankDump output (GPU ranks merge with the same `merge`).
#[cfg(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda"))]
fn run_gpu_command(args: impl Iterator<Item = String>) -> BoxResult<()> {
    use rmc_ab_initio::dump::{same_trajectory, Checkpoint, RankDump};
    use rmc_ab_initio::flat::batched::DEFAULT_MAX_ORDER_GPU;
    use rmc_ab_initio::gpu::run::{run_device, DeviceRunOpts, CYCLES_PER_LAUNCH, DEFAULT_LANES};
    use rmc_ab_initio::gpu::SelectedRuntime;

    let mut options = options(args)?;
    let table_path: PathBuf = take(&mut options, "--table", None)?;
    let mut config: DiagConfig = match options.remove("--config") {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => DiagConfig::default(),
    };
    // 0 = keep the config's chain count (lets wrappers always pass the flag).
    if let Some(chains) = options.remove("--chains") {
        let chains: u64 = chains.parse()?;
        if chains > 0 {
            config.chains = chains;
        }
    }
    let rank: u64 = take(&mut options, "--rank", Some(0))?;
    let lanes: u32 = take(&mut options, "--lanes", Some(DEFAULT_LANES))?;
    let max_order_gpu: usize =
        take(&mut options, "--max-order-gpu", Some(DEFAULT_MAX_ORDER_GPU))?;
    let cycles_per_launch: u64 =
        take(&mut options, "--cycles-per-launch", Some(CYCLES_PER_LAUNCH))?;
    let leg_cycles: u64 = take(&mut options, "--leg-cycles", Some(0))?;
    // Mandatory: every run is named. An anonymous run cannot be resumed,
    // merged, or told apart from an accidental identical resubmission.
    let out_dir: PathBuf = take(&mut options, "--out-dir", None)?;
    if let Some(unknown) = options.keys().next() {
        return Err(format!("unknown option {unknown}").into());
    }

    let table = Arc::new(VertexTable::load(table_path)?);
    let model = Arc::new(AbInitioModel::new(table));
    let chain_offset = rank * config.chains;

    // Resume when this out-dir already holds our checkpoint (the RNG is
    // counter-addressable, so the continued trajectory is bit-identical to an
    // uninterrupted run); otherwise this is leg 0 of a fresh run.
    std::fs::create_dir_all(&out_dir)?;
    let checkpoint_path = out_dir.join(Checkpoint::filename(rank));
    let (resume, leg) = if checkpoint_path.exists() {
        let checkpoint = Checkpoint::load(&checkpoint_path)?;
        // max_steps may grow between legs (extending the schedule is exact:
        // step keys are absolute); everything else must match.
        if !same_trajectory(&checkpoint.config, &config) {
            return Err(
                "checkpoint config differs from --config (only max_steps may change); \
                 refusing to resume"
                    .into(),
            );
        }
        if checkpoint.chain_offset != chain_offset {
            return Err("checkpoint chain offset differs from --rank; refusing to resume".into());
        }
        if checkpoint.config.max_steps != config.max_steps {
            // Extension is exact: whole-cycle schedules only (validated by the
            // driver), so a longer schedule is a strict superset.
            log::info!(
                "extending schedule: max_steps {} → {}",
                checkpoint.config.max_steps,
                config.max_steps
            );
        }
        let leg = checkpoint.leg + 1;
        log::info!(
            "resuming leg {leg} from cycle {} ({})",
            checkpoint.cycles_done,
            checkpoint_path.display()
        );
        (Some((checkpoint.cycles_done, checkpoint.chains)), leg)
    } else {
        (None, 0)
    };

    let deadline = rmc_diagmc::run::slurm_deadline(config.end_time_margin_secs)?;
    let client = <SelectedRuntime as cubecl::prelude::Runtime>::client(&Default::default());
    let started = Instant::now();
    let output = run_device(
        client,
        &model,
        &config,
        DeviceRunOpts {
            chain_offset,
            max_order_gpu,
            lanes,
            cycles_per_launch,
            deadline,
            resume,
            leg_cycles: (leg_cycles > 0).then_some(leg_cycles),
        },
    )?;
    let wall_secs = started.elapsed().as_secs_f64();

    let dump_path = out_dir.join(RankDump::filename(rank, leg));
    RankDump {
        electron_energy_pext: model.electron_energy(config.p_ext),
        chains: config.chains,
        steps_done: output.steps_done,
        data: output.data,
        config: config.clone(),
        rank,
        leg,
    }
    .save(&dump_path)?;
    log::info!("wrote {}", dump_path.display());

    let spc = config.steps_per_cycle.max(1);
    let total_cycles = config.warmup_steps.div_ceil(spc) + config.max_steps.div_ceil(spc);
    let complete = output.cycles_done >= total_cycles;
    Checkpoint {
        chain_offset,
        cycles_done: output.cycles_done,
        leg,
        chains: output.chains.clone(),
        config,
    }
    .save(&checkpoint_path)?;
    log::info!("wrote {}", checkpoint_path.display());

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "report": output.report,
            "update_stats": output.update_stats,
            "step_secs": output.step_secs,
            "replay_secs": output.replay_secs,
            "wall_secs": wall_secs,
            "steps_per_sec_kernel_only": output.steps_done as f64 / output.step_secs,
            "leg": leg,
            "cycles_done": output.cycles_done,
            "total_cycles": total_cycles,
            "complete": complete,
            "out_dir": out_dir,
        }))?
    );
    Ok(())
}

#[cfg(not(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda")))]
fn run_gpu_command(_args: impl Iterator<Item = String>) -> BoxResult<()> {
    Err("run-gpu needs a GPU runtime; rebuild with --features gpu-cuda (or gpu-cpu)".into())
}

/// Phase A1 vertex-dot microbenchmark (plan `gpu-abinitio`).
#[cfg(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda"))]
fn vbench_command(args: impl Iterator<Item = String>) -> BoxResult<()> {
    let mut options = options(args)?;
    let table_path: PathBuf = take(&mut options, "--table", None)?;
    let n_evals: usize = take(&mut options, "--evals", Some(1 << 22))?;
    let seed: u64 = take(&mut options, "--seed", Some(7))?;
    let variants = rmc_ab_initio::gpu::vertex_bench::Variant::parse_list(
        &options
            .remove("--variants")
            .unwrap_or_else(|| "scalar,vector,coop".to_string()),
    )?;
    if let Some(unknown) = options.keys().next() {
        return Err(format!("unknown option {unknown}").into());
    }
    let table = VertexTable::load(table_path)?;
    rmc_ab_initio::gpu::vertex_bench::run_vertex_bench(&table, n_evals, seed, &variants)
}

#[cfg(not(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda")))]
fn vbench_command(_args: impl Iterator<Item = String>) -> BoxResult<()> {
    Err("vbench needs a GPU runtime; rebuild with --features gpu-cuda (or gpu-cpu)".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("def") => {
            let mut config = AbInitioConfig::default();
            if let Some(max_bytes) = env_table_max_bytes()? {
                config.table_max_bytes = max_bytes;
            }
            println!("{}", config.to_json_string()?);
        }
        Some("table") => table_command(args)?,
        Some("run") => table_run_command(args)?,
        Some("run-flat") => run_flat_command(args)?,
        Some("run-gpu") => run_gpu_command(args)?,
        Some("merge") => merge_command(args)?,
        Some("vbench") => vbench_command(args)?,
        Some("model") => {
            let dataset = args
                .next()
                .ok_or("usage: model <dataset_dir> <out.json> [grid_n]")?;
            let out = args
                .next()
                .ok_or("usage: model <dataset_dir> <out.json> [grid_n]")?;
            let n: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(8);
            let m = build_reduced(Path::new(&dataset), n)?;
            m.save(&out)?;
            println!(
                "wrote {out}: {} q-points, omega_LO={:.1} meV",
                m.q.len(),
                m.omega_lo_mev
            );
        }
        Some("alpha") => {
            let cfg = match args.next() {
                Some(path) => load_config(path)?,
                None => AbInitioConfig::default(),
            };
            println!("alpha_LiF: {:.4}", cfg.material.frohlich_alpha());
        }
        Some(path) => {
            let cfg = load_config(path)?;
            let dir = args
                .next()
                .map_or_else(|| PathBuf::from("results-abinitio"), PathBuf::from);
            let report = run_and_write(&cfg, &dir)?;
            println!("{}", report.text());
            println!("results_dir: {}", dir.display());
        }
        None => {
            let cfg = AbInitioConfig::default();
            let report = run_and_write(&cfg, "results-abinitio")?;
            println!("{}", report.text());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_cli_parses_values_and_rejects_bad_memory_limits() {
        let mut parsed = options(
            ["--grid", "50", "--precision", "f32"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(take::<usize>(&mut parsed, "--grid", None).unwrap(), 50);
        assert_eq!(
            take::<TablePrecision>(&mut parsed, "--precision", None).unwrap(),
            TablePrecision::F32
        );
        assert_eq!(gib_to_bytes(1.5).unwrap(), 3 * 1024_u64.pow(3) / 2);
        assert!(gib_to_bytes(0.0).is_err());
        assert!(gib_to_bytes(f64::NAN).is_err());
    }
}
