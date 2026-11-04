use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rmc_ab_initio::config::AbInitioConfig;
use rmc_ab_initio::eph_svd::{DEFAULT_TOL, EphSvd};
use rmc_ab_initio::epw::{Crystal, EpmatWp, EpwData, Wigner};
use rmc_ab_initio::flat::FlatDiagram;
use rmc_ab_initio::flat::batched::BatchedRunOutput;
use rmc_ab_initio::model::{ReducedModel, regular_grid};
use rmc_ab_initio::model_abinitio::AbInitioModel;
use rmc_ab_initio::reconstruct::GammaModel;
use rmc_ab_initio::run::run_and_write;
use rmc_ab_initio::vertex_table::{
    DEFAULT_MAX_BYTES, DEFAULT_PHFREQ_CUTOFF_EV, FillDevice, ShardMeta, TablePrecision, VertexTable,
};
use rmc_core::random::DefaultRng;
use rmc_diagmc::measure::MeasurementData;
use rmc_diagmc::{DiagConfig, Diagram, Model};

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

/// Parse a `--p-ext x,y,z` override (fractional coordinates; must be a point
/// of the table's momentum grid).
fn parse_p_ext(arg: &str) -> BoxResult<nalgebra::Vector3<f64>> {
    let parts = arg
        .split(',')
        .map(|part| part.trim().parse::<f64>())
        .collect::<Result<Vec<f64>, _>>()
        .map_err(|error| format!("invalid --p-ext value '{arg}': {error}"))?;
    match parts.as_slice() {
        [x, y, z] => Ok(nalgebra::Vector3::new(*x, *y, *z)),
        _ => Err(
            format!("--p-ext needs exactly three comma-separated components, got '{arg}'").into(),
        ),
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
    // Base data is always CPU; --device gpu moves the Fourier fill to the GPU.
    let device: FillDevice = take(&mut options, "--device", Some(FillDevice::Cpu))?;
    // $RMC_GPU_CHUNK: points per GPU launch; lower it if the device OOMs (the B
    // intermediate is chunk·nmodes·rank complex f64).
    let device = match (device, std::env::var("RMC_GPU_CHUNK")) {
        (FillDevice::Gpu { .. }, Ok(value)) => FillDevice::Gpu {
            chunk_points: value
                .parse()
                .map_err(|error| format!("invalid RMC_GPU_CHUNK '{value}': {error}"))?,
        },
        (device, _) => device,
    };
    // Fail GPU misconfigurations here, in milliseconds — build_range checks them
    // too, but only after the dataset load and SVD have burned walltime.
    if matches!(device, FillDevice::Gpu { .. }) {
        if precision != TablePrecision::F32 {
            return Err("GPU fill produces f32 tables only; use --precision f32".into());
        }
        if cfg!(not(any(
            feature = "gpu-cpu",
            feature = "gpu-hip",
            feature = "gpu-cuda"
        ))) {
            return Err(
                "GPU fill needs a GPU runtime: build with --features gpu-cuda (or gpu-cpu)".into(),
            );
        }
    }
    // A shard fills only slab [i*nk/S, (i+1)*nk/S) and defers the global
    // band-edge shift to `table-merge`; --shard and --shards go together.
    let shard: Option<usize> = options
        .remove("--shard")
        .map(|v| {
            v.parse()
                .map_err(|e| format!("invalid --shard value '{v}': {e}"))
        })
        .transpose()?;
    let shards: Option<usize> = options
        .remove("--shards")
        .map(|v| {
            v.parse()
                .map_err(|e| format!("invalid --shards value '{v}': {e}"))
        })
        .transpose()?;
    let out = options.remove("--out").map_or_else(
        || dataset.join(VertexTable::cache_filename(grid_n, total_rank, precision)),
        PathBuf::from,
    );
    if let Some(unknown) = options.keys().next() {
        return Err(format!("unknown option {unknown}").into());
    }

    let (shard, shards) = match (shard, shards) {
        (None, None) => {
            build_full_table(
                &dataset, grid_n, total_rank, precision, max_bytes, device, &out,
            )?;
            return Ok(());
        }
        (Some(i), Some(s)) => (i, s),
        _ => return Err("--shard and --shards must be given together".into()),
    };
    if shards == 0 || shard >= shards {
        return Err(format!("--shard {shard} out of range for --shards {shards}").into());
    }
    let nk = grid_n.checked_pow(3).ok_or("grid cubed overflows usize")?;
    let (lo, hi) = (shard * nk / shards, (shard + 1) * nk / shards);
    let meta = ShardMeta {
        grid_n,
        shards,
        shard,
        lo,
        hi,
        total_rank,
        precision,
    };
    let shard_out = {
        let mut name = out.into_os_string();
        name.push(format!(".shard-{shard}-of-{shards}"));
        PathBuf::from(name)
    };

    // Resume: a matching shard on disk means this leg's work is already done —
    // resubmitting the devel job is a plain no-op for finished shards.
    if shard_out.exists() {
        if let Ok(existing) = VertexTable::read_shard_meta(&shard_out) {
            if existing == meta {
                println!(
                    "shard {shard}/{shards} already built: {}",
                    shard_out.display()
                );
                return Ok(());
            }
        }
    }

    let (epw, crystal, wigner, epmat) = load_vertex_inputs(&dataset)?;
    let started = Instant::now();
    let svd = EphSvd::compress_global(&epmat, DEFAULT_TOL, total_rank);
    println!("compression time: {:.3} s", started.elapsed().as_secs_f64());
    println!(
        "shard {shard}/{shards}: points [{lo}, {hi}) of {nk} (components={})",
        svd.rank()
    );
    println!("output: {}", shard_out.display());
    let started = Instant::now();
    let table = VertexTable::build_range(
        &epw,
        &crystal,
        &wigner,
        &svd,
        grid_n,
        lo,
        hi,
        DEFAULT_PHFREQ_CUTOFF_EV,
        precision,
        max_bytes,
        false,
        device,
    )?;
    println!("build time: {:.3} s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    table.save_shard(&meta, &shard_out)?;
    println!(
        "wrote {} in {:.3} s",
        shard_out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Full-grid build (the non-sharded path): SVD, build, save.
fn build_full_table(
    dataset: &Path,
    grid_n: usize,
    total_rank: usize,
    precision: TablePrecision,
    max_bytes: u64,
    device: FillDevice,
    out: &Path,
) -> BoxResult<()> {
    let (epw, crystal, wigner, epmat) = load_vertex_inputs(dataset)?;
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
    let nk = grid_n.checked_pow(3).ok_or("grid cubed overflows usize")?;
    let started = Instant::now();
    let table = VertexTable::build_range(
        &epw,
        &crystal,
        &wigner,
        &svd,
        grid_n,
        0,
        nk,
        DEFAULT_PHFREQ_CUTOFF_EV,
        precision,
        max_bytes,
        true,
        device,
    )?;
    println!("build time: {:.3} s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    table.save(out)?;
    println!(
        "wrote {} in {:.3} s",
        out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// `table-merge --out <final.rmcvtab> <shards...>`: reassemble a full table from
/// its complete shard set (validates the set, applies the global band-edge shift).
fn table_merge_command(args: impl Iterator<Item = String>) -> BoxResult<()> {
    let mut out: Option<PathBuf> = None;
    let mut paths = Vec::new();
    let mut args = args;
    while let Some(arg) = args.next() {
        if arg == "--out" {
            out = Some(PathBuf::from(args.next().ok_or("missing value for --out")?));
        } else if arg.starts_with("--") {
            return Err(format!("unknown option {arg} (only --out is accepted)").into());
        } else {
            paths.push(PathBuf::from(arg));
        }
    }
    let out = out.ok_or("usage: table-merge --out <final.rmcvtab> <shards...>")?;
    if paths.is_empty() {
        return Err("no shard files given".into());
    }
    let shards = paths
        .iter()
        .map(VertexTable::load_shard)
        .collect::<Result<Vec<_>, _>>()?;
    let started = Instant::now();
    let table = VertexTable::merge_shards(shards)?;
    table.save(&out)?;
    println!(
        "merged {} shards → {} in {:.3} s",
        paths.len(),
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

/// One-line human summary of a report — job logs carry log lines, not data.
fn report_summary(report: &rmc_diagmc::Report) -> String {
    format!(
        "e0_thermo {:.5} ± {:.5} eV, order {:.2}, sign {:.4}, {:.3e} steps",
        report.e0_thermo.mean,
        report.e0_thermo.stderr,
        report.mean_order.mean,
        report.sign.mean,
        report.steps_done as f64
    )
}

/// Load a rank's checkpoint from `out_dir` if present, validating it against the
/// current config (only `max_steps` may grow between legs — the schedule is a
/// strict superset because step keys are absolute). Returns the resume point
/// `(cycles_done, chains)` and the new leg index, or `(None, 0)` for a fresh
/// run. Shared by `run-cpu` and `run-gpu`.
fn load_resume(
    out_dir: &Path,
    rank: u64,
    config: &DiagConfig,
    chain_offset: u64,
) -> BoxResult<(Option<(u64, Vec<FlatDiagram>)>, u64)> {
    use rmc_ab_initio::dump::{Checkpoint, same_trajectory};

    let checkpoint_path = out_dir.join(Checkpoint::filename(rank));
    if !checkpoint_path.exists() {
        return Ok((None, 0));
    }
    let checkpoint = Checkpoint::load(&checkpoint_path)?;
    // max_steps may grow between legs (extending the schedule is exact: step
    // keys are absolute); everything else must match.
    if !same_trajectory(&checkpoint.config, config) {
        return Err(
            "checkpoint config differs from --config (only max_steps may change); \
             refusing to resume"
                .into(),
        );
    }
    if checkpoint.chain_offset != chain_offset {
        return Err("checkpoint chain offset differs from --rank; refusing to resume".into());
    }
    // Provenance note, not a guard: reproducing a run means rebuilding at the
    // recorded hash (`slurm submit --commit-hash`); mixed-code legs are allowed.
    if checkpoint.code != rmc_ab_initio::code_version() {
        log::warn!(
            "code drift: checkpoint from {}, this binary is {}",
            checkpoint.code,
            rmc_ab_initio::code_version()
        );
    }
    if checkpoint.config.max_steps != config.max_steps {
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
    Ok((Some((checkpoint.cycles_done, checkpoint.chains)), leg))
}

/// Write `run-metadata.json` into `out_dir`: ONLY what no other run artifact
/// records — the identity of the vertex table (grid, kept SVD components,
/// filename, read from the table file itself, not the CLI) and the engine.
/// Deliberately nothing more: the run dir already owns its `DiagConfig` (in
/// the checkpoint and the job dir's config copy) and its per-leg code
/// provenance (in every result JSON, dump, and checkpoint).
///
/// An existing file is immutable: every rank checks it and refuses to continue
/// a run whose recorded table or engine differs. This is a physics guard, not
/// bookkeeping — the checkpoint's `same_trajectory` pins the config across
/// legs but records nothing about the table, so without this check a resume
/// against the wrong table would silently mix estimators. Rank 0 creates the
/// file (via temp + rename, so a concurrent rank never reads a torn file).
fn write_run_metadata(
    out_dir: &Path,
    rank: u64,
    engine: &str,
    table_path: &Path,
    grid_n: usize,
    svd_rank: usize,
    // The external momentum this run samples (fractional coordinates). `None`
    // only in metadata-only mode without an explicit --p-ext, whose config's
    // p_ext is a placeholder that must not be recorded.
    p_ext: Option<nalgebra::Vector3<f64>>,
) -> BoxResult<()> {
    let mut metadata = serde_json::json!({
        "engine": engine,
        "grid": grid_n,
        // Components actually kept by the SVD (the filename's r<N> is the cap).
        "svd_rank": svd_rank,
        "table": table_path.file_name().map(|name| name.to_string_lossy()),
    });
    if let Some(p) = p_ext {
        metadata["p_ext"] = serde_json::json!([p.x, p.y, p.z]);
    }
    let path = out_dir.join("run-metadata.json");
    if path.exists() {
        let recorded: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        // Files written before p_ext was recorded (or by metadata-only mode
        // without --p-ext) are compared without it and then upgraded in place;
        // a recorded p_ext must match exactly.
        let mut expected = metadata.clone();
        if recorded.get("p_ext").is_none() || p_ext.is_none() {
            expected.as_object_mut().map(|m| m.remove("p_ext"));
        }
        let mut recorded_cmp = recorded.clone();
        if p_ext.is_none() {
            recorded_cmp.as_object_mut().map(|m| m.remove("p_ext"));
        }
        if recorded_cmp != expected {
            return Err(format!(
                "{} does not match this invocation: recorded {recorded}, this run would \
                 write {metadata}; refusing to continue a run with a different table, engine, \
                 or external momentum",
                path.display()
            )
            .into());
        }
        if recorded == metadata || p_ext.is_none() || rank != 0 {
            return Ok(());
        }
    } else if rank != 0 {
        return Ok(());
    }
    std::fs::create_dir_all(out_dir)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&metadata)?)?;
    std::fs::rename(&tmp, &path)?;
    log::info!("wrote {}", path.display());
    Ok(())
}

/// Save one leg's per-rank `RankDump` (this leg's raw `MeasurementData`) to
/// `out_dir`. Engine-agnostic — both CPU engines and `run-gpu` produce the same
/// `MeasurementData`, and `merge` consumes these dumps.
fn write_rank_dump(
    out_dir: &Path,
    rank: u64,
    leg: u64,
    config: &DiagConfig,
    model: &AbInitioModel,
    data: &MeasurementData,
    steps_done: u64,
) -> BoxResult<()> {
    use rmc_ab_initio::dump::RankDump;
    let dump_path = out_dir.join(RankDump::filename(rank, leg));
    RankDump {
        electron_energy_pext: model.electron_energy(config.p_ext),
        chains: config.chains,
        steps_done,
        data: data.clone(),
        config: config.clone(),
        rank,
        leg,
        code: rmc_ab_initio::code_version().to_string(),
    }
    .save(&dump_path)?;
    log::info!("wrote {}", dump_path.display());
    Ok(())
}

/// Write one leg's `RankDump` + batched `Checkpoint` + result JSON to `out_dir`
/// and print the closing summary. `extra` merges engine-specific fields into the
/// result JSON. Used by the batched `run-flat` and `run-gpu`.
#[allow(clippy::too_many_arguments)]
fn write_leg_outputs(
    out_dir: &Path,
    rank: u64,
    leg: u64,
    config: &DiagConfig,
    chain_offset: u64,
    model: &AbInitioModel,
    output: &BatchedRunOutput,
    wall_secs: f64,
    extra: serde_json::Value,
) -> BoxResult<()> {
    use rmc_ab_initio::dump::Checkpoint;

    // A warmup-only leg (deadline hit inside warmup) has no samples: nothing
    // to dump for merge; the checkpoint below still records its progress.
    if output.steps_done > 0 {
        write_rank_dump(
            out_dir,
            rank,
            leg,
            config,
            model,
            &output.data,
            output.steps_done,
        )?;
    }

    let spc = config.steps_per_cycle.max(1);
    let total_cycles = config.warmup_steps.div_ceil(spc) + config.max_steps.div_ceil(spc);
    let complete = output.cycles_done >= total_cycles;
    let checkpoint_path = out_dir.join(Checkpoint::filename(rank));
    Checkpoint {
        chain_offset,
        cycles_done: output.cycles_done,
        leg,
        chains: output.chains.clone(),
        config: config.clone(),
        code: rmc_ab_initio::code_version().to_string(),
    }
    .save(&checkpoint_path)?;
    log::info!("wrote {}", checkpoint_path.display());

    // Data goes to the run dir; stdout carries only the closing summary line.
    let mut result = serde_json::json!({
        "report": output.report,
        "update_stats": output.update_stats,
        "step_secs": output.step_secs,
        "replay_secs": output.replay_secs,
        "wall_secs": wall_secs,
        "leg": leg,
        "cycles_done": output.cycles_done,
        "total_cycles": total_cycles,
        "complete": complete,
        "out_dir": out_dir,
        "code": rmc_ab_initio::code_version(),
    });
    if let (Some(base), Some(add)) = (result.as_object_mut(), extra.as_object()) {
        for (key, value) in add {
            base.insert(key.clone(), value.clone());
        }
    }
    let result_path = out_dir.join(format!("result-rank{rank}-leg{leg}.json"));
    std::fs::write(&result_path, serde_json::to_string_pretty(&result)?)?;
    let summary = match &output.report {
        Some(report) => report_summary(report),
        None => "warmup-only leg (no measured samples yet)".to_string(),
    };
    println!(
        "rank {rank} leg {leg}: {summary}, cycles {}/{total_cycles}{}, {:.3e} steps/s [{}]",
        output.cycles_done,
        if complete { ", complete" } else { "" },
        output.steps_done as f64 / output.step_secs,
        result_path.display()
    );
    Ok(())
}

/// `run-flat`: the batched `FlatDiagram` CPU engine — the GPU-parity reference
/// (CPU twin of `run-gpu`), kept even though it's ~20× slower per core than the
/// slotmap engine that now backs `run-cpu` (its `FlatDiagram` does O(order)
/// updates for GPU-friendliness). Same `RankDump`/`Checkpoint`/`merge` machinery.
fn run_flat_command(args: impl Iterator<Item = String>) -> BoxResult<()> {
    use rmc_ab_initio::flat::batched::{BatchedRunOpts, DEFAULT_MAX_ORDER_GPU, run_batched_opts};

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
    // Empty = keep the config's p_ext (lets wrappers always pass the flag).
    // A first-class override like --chains: the config the engine runs — and
    // the checkpoint records, and same_trajectory pins across legs — is the
    // post-override one; no patched temp config anywhere.
    let p_ext_given = match options.remove("--p-ext") {
        Some(p_ext) if !p_ext.is_empty() => {
            config.p_ext = parse_p_ext(&p_ext)?;
            true
        }
        _ => false,
    };
    let rank: u64 = take(&mut options, "--rank", Some(0))?;
    let max_order_gpu: usize = take(&mut options, "--max-order-gpu", Some(DEFAULT_MAX_ORDER_GPU))?;
    let leg_cycles: u64 = take(&mut options, "--leg-cycles", Some(0))?;
    // Mandatory: every run is named. An anonymous run cannot be resumed,
    // merged, or told apart from an accidental identical resubmission.
    let out_dir: PathBuf = take(&mut options, "--out-dir", None)?;
    if let Some(unknown) = options.keys().next() {
        return Err(format!("unknown option {unknown}").into());
    }

    // Metadata-only mode — see the run-cpu twin of this block.
    if config.max_steps == 0 {
        let head = VertexTable::load_head(&table_path)?;
        return write_run_metadata(
            &out_dir,
            rank,
            "cpu-flat",
            &table_path,
            head.grid_n,
            head.s.len(),
            p_ext_given.then_some(config.p_ext),
        );
    }

    let table = Arc::new(VertexTable::load(&table_path)?);
    write_run_metadata(
        &out_dir,
        rank,
        "cpu-flat",
        &table_path,
        table.grid_n,
        table.s.len(),
        Some(config.p_ext),
    )?;
    let model = Arc::new(AbInitioModel::new(table));
    let chain_offset = rank * config.chains;

    // Resume when this out-dir already holds our checkpoint; otherwise leg 0.
    std::fs::create_dir_all(&out_dir)?;
    let (resume, leg) = load_resume(&out_dir, rank, &config, chain_offset)?;

    let deadline = rmc_diagmc::run::slurm_deadline(config.end_time_margin_secs)?;
    let started = Instant::now();
    let output = run_batched_opts(
        &model,
        &config,
        BatchedRunOpts {
            chain_offset,
            max_order_gpu,
            deadline,
            resume,
            leg_cycles: (leg_cycles > 0).then_some(leg_cycles),
        },
    )?;
    let wall_secs = started.elapsed().as_secs_f64();

    write_leg_outputs(
        &out_dir,
        rank,
        leg,
        &config,
        chain_offset,
        &model,
        &output,
        wall_secs,
        serde_json::json!({
            "steps_per_sec_step_loop": output.steps_done as f64 / output.step_secs,
        }),
    )?;
    Ok(())
}

/// Load a rank's `SlotmapCheckpoint` from `out_dir` if present, validating it
/// against the current config (only `max_steps` may change between legs).
/// Returns each chain's saved `(Diagram, rng)` and the new leg index, or
/// `(None, 0)` for a fresh run. The slotmap analog of [`load_resume`].
fn load_slotmap_resume(
    out_dir: &Path,
    rank: u64,
    config: &DiagConfig,
    chain_offset: u64,
) -> BoxResult<(Option<Vec<(Diagram, DefaultRng)>>, u64)> {
    use rmc_ab_initio::dump::{SlotmapCheckpoint, same_trajectory};

    let checkpoint_path = out_dir.join(SlotmapCheckpoint::filename(rank));
    if !checkpoint_path.exists() {
        return Ok((None, 0));
    }
    let checkpoint = SlotmapCheckpoint::load(&checkpoint_path)?;
    if !same_trajectory(&checkpoint.config, config) {
        return Err(
            "checkpoint config differs from --config (only max_steps may change); \
             refusing to resume"
                .into(),
        );
    }
    if checkpoint.chain_offset != chain_offset {
        return Err("checkpoint chain offset differs from --rank; refusing to resume".into());
    }
    // Provenance note, not a guard: reproducing a run means rebuilding at the
    // recorded hash (`slurm submit --commit-hash`); mixed-code legs are allowed.
    if checkpoint.code != rmc_ab_initio::code_version() {
        log::warn!(
            "code drift: checkpoint from {}, this binary is {}",
            checkpoint.code,
            rmc_ab_initio::code_version()
        );
    }
    let leg = checkpoint.leg + 1;
    log::info!(
        "resuming leg {leg} from {} ({} chains)",
        checkpoint_path.display(),
        checkpoint.chains.len()
    );
    Ok((Some(checkpoint.chains), leg))
}

/// `run-cpu`: the slotmap CPU engine — the fast production path (O(1) diagram
/// updates, ~20× the per-core throughput of `run-flat`). Checkpoint + multi-node
/// like `run-gpu`: resuming from a `SlotmapCheckpoint` in `--out-dir` continues
/// each chain's exact trajectory (bit-identical up to the drift-check resync,
/// whose cadence restarts per leg — see `SlotmapCheckpoint`), and independent
/// `--rank`s across nodes merge offline. Each invocation runs one leg (up to
/// `max_steps`/chain or until the deadline); legs accumulate via `merge`.
fn run_cpu_command(args: impl Iterator<Item = String>) -> BoxResult<()> {
    use rmc_ab_initio::dump::SlotmapCheckpoint;

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
    // Empty = keep the config's p_ext (lets wrappers always pass the flag).
    // A first-class override like --chains: the config the engine runs — and
    // the checkpoint records, and same_trajectory pins across legs — is the
    // post-override one; no patched temp config anywhere.
    let p_ext_given = match options.remove("--p-ext") {
        Some(p_ext) if !p_ext.is_empty() => {
            config.p_ext = parse_p_ext(&p_ext)?;
            true
        }
        _ => false,
    };
    let rank: u64 = take(&mut options, "--rank", Some(0))?;
    // Mandatory: every run is named. An anonymous run cannot be resumed, merged,
    // or told apart from an accidental identical resubmission.
    let out_dir: PathBuf = take(&mut options, "--out-dir", None)?;
    if let Some(unknown) = options.keys().next() {
        return Err(format!("unknown option {unknown}").into());
    }

    // max_steps == 0 never samples (`DiagConfig::validate` rejects it), so it is
    // the explicit metadata-only mode (configs/metadata-only.json): annotate an
    // existing run with run-metadata.json and exit, never touching its dumps.
    // Reads only the table head, not the tens-of-GiB payload — login-node safe.
    if config.max_steps == 0 {
        let head = VertexTable::load_head(&table_path)?;
        return write_run_metadata(
            &out_dir,
            rank,
            "cpu-slotmap",
            &table_path,
            head.grid_n,
            head.s.len(),
            p_ext_given.then_some(config.p_ext),
        );
    }

    let table = Arc::new(VertexTable::load(&table_path)?);
    write_run_metadata(
        &out_dir,
        rank,
        "cpu-slotmap",
        &table_path,
        table.grid_n,
        table.s.len(),
        Some(config.p_ext),
    )?;
    let model = Arc::new(AbInitioModel::new(table));
    let chain_offset = rank * config.chains;

    std::fs::create_dir_all(&out_dir)?;
    let (resume, leg) = load_slotmap_resume(&out_dir, rank, &config, chain_offset)?;

    let started = Instant::now();
    let leg_out = rmc_diagmc::run_leg(model.clone(), config.clone(), chain_offset, resume)?;
    let wall_secs = started.elapsed().as_secs_f64();

    // This leg's samples → RankDump (for merge); every chain's final state → the
    // checkpoint (for the next leg).
    write_rank_dump(
        &out_dir,
        rank,
        leg,
        &config,
        &model,
        &leg_out.data,
        leg_out.steps_done,
    )?;
    let checkpoint_path = out_dir.join(SlotmapCheckpoint::filename(rank));
    SlotmapCheckpoint {
        config: config.clone(),
        chain_offset,
        leg,
        code: rmc_ab_initio::code_version().to_string(),
        chains: leg_out.chains,
    }
    .save(&checkpoint_path)?;
    log::info!("wrote {}", checkpoint_path.display());

    let result = serde_json::json!({
        "report": leg_out.report,
        "wall_secs": wall_secs,
        "leg": leg,
        "steps_done": leg_out.steps_done,
        "complete": leg_out.complete,
        "steps_per_sec": leg_out.steps_done as f64 / wall_secs,
        "out_dir": out_dir,
        "code": rmc_ab_initio::code_version(),
    });
    let result_path = out_dir.join(format!("result-rank{rank}-leg{leg}.json"));
    std::fs::write(&result_path, serde_json::to_string_pretty(&result)?)?;
    println!(
        "rank {rank} leg {leg}: {}, {} steps{}, {:.3e} steps/s [{}]",
        report_summary(&leg_out.report),
        leg_out.steps_done,
        if leg_out.complete { ", complete" } else { "" },
        leg_out.steps_done as f64 / wall_secs,
        result_path.display()
    );
    Ok(())
}

/// Merge rank dumps into one report: `merge [--out report.json] <dumps...>`.
fn merge_command(args: impl Iterator<Item = String>) -> BoxResult<()> {
    use rmc_ab_initio::dump::{RankDump, merge_dumps};

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
    match out {
        Some(out) => {
            // Data to the file, one summary line to the log.
            std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
            println!(
                "merged {} dumps: {} [{}]",
                paths.len(),
                report_summary(&report),
                out.display()
            );
        }
        None => println!("{}", serde_json::to_string_pretty(&report)?),
    }
    Ok(())
}

/// Phase A4/A5 device run: `run-cpu` semantics on the selected GPU runtime,
/// same RankDump output (GPU ranks merge with the same `merge`).
#[cfg(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda"))]
fn run_gpu_command(args: impl Iterator<Item = String>) -> BoxResult<()> {
    use rmc_ab_initio::flat::batched::DEFAULT_MAX_ORDER_GPU;
    use rmc_ab_initio::gpu::SelectedRuntime;
    use rmc_ab_initio::gpu::run::{CYCLES_PER_LAUNCH, DEFAULT_LANES, DeviceRunOpts, run_device};

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
    // Empty = keep the config's p_ext (lets wrappers always pass the flag).
    // A first-class override like --chains: the config the engine runs — and
    // the checkpoint records, and same_trajectory pins across legs — is the
    // post-override one; no patched temp config anywhere.
    let p_ext_given = match options.remove("--p-ext") {
        Some(p_ext) if !p_ext.is_empty() => {
            config.p_ext = parse_p_ext(&p_ext)?;
            true
        }
        _ => false,
    };
    let rank: u64 = take(&mut options, "--rank", Some(0))?;
    let lanes: u32 = take(&mut options, "--lanes", Some(DEFAULT_LANES))?;
    let max_order_gpu: usize = take(&mut options, "--max-order-gpu", Some(DEFAULT_MAX_ORDER_GPU))?;
    let cycles_per_launch: u64 =
        take(&mut options, "--cycles-per-launch", Some(CYCLES_PER_LAUNCH))?;
    let leg_cycles: u64 = take(&mut options, "--leg-cycles", Some(0))?;
    // Mandatory: every run is named. An anonymous run cannot be resumed,
    // merged, or told apart from an accidental identical resubmission.
    let out_dir: PathBuf = take(&mut options, "--out-dir", None)?;
    if let Some(unknown) = options.keys().next() {
        return Err(format!("unknown option {unknown}").into());
    }

    // Metadata-only mode — see the run-cpu twin of this block. Returns before
    // any device work, so it runs fine on a node without a GPU.
    if config.max_steps == 0 {
        let head = VertexTable::load_head(&table_path)?;
        return write_run_metadata(
            &out_dir,
            rank,
            "gpu",
            &table_path,
            head.grid_n,
            head.s.len(),
            p_ext_given.then_some(config.p_ext),
        );
    }

    let table = Arc::new(VertexTable::load(&table_path)?);
    write_run_metadata(
        &out_dir,
        rank,
        "gpu",
        &table_path,
        table.grid_n,
        table.s.len(),
        Some(config.p_ext),
    )?;
    let model = Arc::new(AbInitioModel::new(table));
    let chain_offset = rank * config.chains;

    // Resume when this out-dir already holds our checkpoint (the RNG is
    // counter-addressable, so the continued trajectory is bit-identical to an
    // uninterrupted run); otherwise this is leg 0 of a fresh run.
    std::fs::create_dir_all(&out_dir)?;
    let (resume, leg) = load_resume(&out_dir, rank, &config, chain_offset)?;

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

    write_leg_outputs(
        &out_dir,
        rank,
        leg,
        &config,
        chain_offset,
        &model,
        &output,
        wall_secs,
        serde_json::json!({
            "steps_per_sec_kernel_only": output.steps_done as f64 / output.step_secs,
        }),
    )?;
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
        Some("table-merge") => table_merge_command(args)?,
        Some("run") => table_run_command(args)?,
        Some("run-cpu") => run_cpu_command(args)?,
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
