//! Per-rank batch dumps + offline merge (the `multi-node` design, consumed by
//! CPU ranks and GPU ranks alike). A dump carries the raw `MeasurementData`
//! plus the one model scalar `summarize` needs, so merging never reloads a
//! vertex table. Merging sorts by rank and concatenates batch lists in that
//! fixed order — the merged report is a pure function of (seed, total chains).

use std::path::Path;

use rmc_core::Merge;
use rmc_diagmc::measure::{summarize, MeasurementData};
use rmc_diagmc::{DiagConfig, Report};
use serde::{Deserialize, Serialize};

type BoxResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Serialize, Deserialize)]
pub struct RankDump {
    pub config: DiagConfig,
    pub rank: u64,
    pub chains: u64,
    pub electron_energy_pext: f64,
    pub steps_done: u64,
    pub data: MeasurementData,
}

impl RankDump {
    pub fn filename(rank: u64) -> String {
        format!("batches-rank{rank}.bin")
    }

    pub fn save(&self, path: impl AsRef<Path>) -> BoxResult<()> {
        let file = std::fs::File::create(path)?;
        bincode::serialize_into(std::io::BufWriter::new(file), self)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> BoxResult<Self> {
        let file = std::fs::File::open(&path)
            .map_err(|e| format!("cannot open dump '{}': {e}", path.as_ref().display()))?;
        Ok(bincode::deserialize_from(std::io::BufReader::new(file))?)
    }
}

/// Validate, sort by rank, merge in rank order, summarize.
pub fn merge_dumps(mut dumps: Vec<RankDump>) -> BoxResult<Report> {
    if dumps.is_empty() {
        return Err("no dumps to merge".into());
    }
    dumps.sort_by_key(|dump| dump.rank);
    let first = &dumps[0];
    for dump in &dumps[1..] {
        if dump.config != first.config {
            return Err(format!("rank {} config differs from rank {}", dump.rank, first.rank).into());
        }
        if dump.electron_energy_pext != first.electron_energy_pext {
            return Err(format!("rank {} ε(p_ext) differs", dump.rank).into());
        }
    }
    for pair in dumps.windows(2) {
        if pair[0].rank == pair[1].rank {
            return Err(format!("duplicate rank {}", pair[0].rank).into());
        }
        if pair[1].rank != pair[0].rank + 1 {
            log::warn!(
                "rank gap between {} and {} — missing statistics, estimate still valid",
                pair[0].rank,
                pair[1].rank
            );
        }
    }
    let config = first.config.clone();
    let electron_energy_pext = first.electron_energy_pext;
    let steps_done: u64 = dumps.iter().map(|dump| dump.steps_done).sum();
    let data = dumps
        .into_iter()
        .map(|dump| dump.data)
        .reduce(Merge::merge)
        .expect("non-empty");
    let mut report = summarize(
        data,
        electron_energy_pext,
        config.mu,
        config.max_tau,
        config.tau_fit,
        config.fixed_tau,
    )?;
    report.steps_done = steps_done;
    Ok(report)
}
