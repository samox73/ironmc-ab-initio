use std::path::{Path, PathBuf};

use rmc_frohlich::app::{run_from_config_with_progress, write_results};
use rmc_frohlich::config::RunConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("def") => {
            println!("{}", RunConfig::default().to_json_string()?);
        }
        Some("bench") => {
            let cfg = match args.next() {
                Some(path) => RunConfig::load_json(path)?,
                None if Path::new("input.json").exists() => RunConfig::load_json("input.json")?,
                None => RunConfig::default(),
            };
            let report = rmc_frohlich::app::run_bench(&cfg)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            println!("steps/sec: {:.3}", report.steps_per_sec);
        }
        Some("gpu") => {
            #[cfg(feature = "gpu")]
            {
                let cfg = match args.next() {
                    Some(path) => RunConfig::load_json(path)?,
                    None if Path::new("input.json").exists() => RunConfig::load_json("input.json")?,
                    None => RunConfig::default(),
                };
                let output = rmc_frohlich::gpu::run_gpu_from_config(&cfg)?;
                let results_dir = args
                    .next()
                    .map_or_else(|| PathBuf::from("results-gpu"), PathBuf::from);
                let manifest = write_results(&cfg, &output, &results_dir)?;
                println!("{}", manifest.summary.text());
                println!("results_dir: {}", results_dir.display());
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err("the 'gpu' subcommand requires building with --features gpu,gpu-cpu, gpu,gpu-hip, or gpu,gpu-cuda".into());
            }
        }
        Some(path) => {
            let cfg = RunConfig::load_json(path)?;
            let output = run_from_config_with_progress(&cfg, true)?;
            let results_dir = args
                .next()
                .map_or_else(|| PathBuf::from("results"), PathBuf::from);
            let manifest = write_results(&cfg, &output, &results_dir)?;
            println!();
            println!("{}", manifest.summary.text());
            println!("results_dir: {}", results_dir.display());
        }
        None => {
            let cfg = RunConfig::load_json("input.json").unwrap_or_default();
            let output = run_from_config_with_progress(&cfg, true)?;
            let results_dir = PathBuf::from("results");
            let manifest = write_results(&cfg, &output, &results_dir)?;
            println!();
            println!("{}", manifest.summary.text());
            println!("results_dir: {}", results_dir.display());
        }
    }
    Ok(())
}
