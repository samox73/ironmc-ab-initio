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

// Versioned file headers (the VertexTable::save idiom): adding fields to
// these bincode payloads is a version bump, not silent data orphaning.
const DUMP_MAGIC: &[u8; 8] = b"RMCDUMP\0";
const CKPT_MAGIC: &[u8; 8] = b"RMCCKPT\0";
const FORMAT_VERSION: u32 = 1;

fn write_header(writer: &mut impl std::io::Write, magic: &[u8; 8]) -> BoxResult<()> {
    writer.write_all(magic)?;
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
    Ok(())
}

fn check_header(reader: &mut impl std::io::Read, magic: &[u8; 8], what: &str) -> BoxResult<()> {
    let mut file_magic = [0u8; 8];
    let mut version = [0u8; 4];
    reader.read_exact(&mut file_magic)?;
    reader.read_exact(&mut version)?;
    if &file_magic != magic {
        return Err(format!("not a {what} file (bad magic); regenerate").into());
    }
    let version = u32::from_le_bytes(version);
    if version != FORMAT_VERSION {
        return Err(format!(
            "{what} format version {version} is unsupported (expected {FORMAT_VERSION}); regenerate"
        )
        .into());
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
pub struct RankDump {
    pub config: DiagConfig,
    pub rank: u64,
    /// Checkpoint leg this dump belongs to (0 for single-leg runs). Legs of
    /// one rank carry disjoint slices of the same chains' trajectories.
    pub leg: u64,
    /// Build provenance of the producing binary (`crate::code_version()`).
    pub code: String,
    pub chains: u64,
    pub electron_energy_pext: f64,
    pub steps_done: u64,
    pub data: MeasurementData,
}

impl RankDump {
    /// Leg 0 keeps the historical name so single-leg workflows and the
    /// `batches-rank*.bin` globs are unchanged.
    pub fn filename(rank: u64, leg: u64) -> String {
        if leg == 0 {
            format!("batches-rank{rank}.bin")
        } else {
            format!("batches-rank{rank}-leg{leg}.bin")
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> BoxResult<()> {
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        write_header(&mut writer, DUMP_MAGIC)?;
        bincode::serialize_into(&mut writer, self)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> BoxResult<Self> {
        let file = std::fs::File::open(&path)
            .map_err(|e| format!("cannot open dump '{}': {e}", path.as_ref().display()))?;
        let mut reader = std::io::BufReader::new(file);
        check_header(&mut reader, DUMP_MAGIC, "RankDump")?;
        Ok(bincode::deserialize_from(&mut reader)?)
    }
}

/// Same physical trajectory: config equality ignoring `max_steps`, the one
/// field a checkpointed run may legally extend between legs (step keys are
/// schedule-independent, so a longer schedule is a superset).
pub fn same_trajectory(a: &DiagConfig, b: &DiagConfig) -> bool {
    let mut a = a.clone();
    a.max_steps = b.max_steps;
    a == *b
}

/// Validate, sort by (rank, leg), merge in that fixed order, summarize.
pub fn merge_dumps(mut dumps: Vec<RankDump>) -> BoxResult<Report> {
    if dumps.is_empty() {
        return Err("no dumps to merge".into());
    }
    dumps.sort_by_key(|dump| (dump.rank, dump.leg));
    let first = &dumps[0];
    for dump in &dumps[1..] {
        if !same_trajectory(&dump.config, &first.config) {
            return Err(format!("rank {} config differs from rank {}", dump.rank, first.rank).into());
        }
        if dump.electron_energy_pext != first.electron_energy_pext {
            return Err(format!("rank {} ε(p_ext) differs", dump.rank).into());
        }
    }
    for pair in dumps.windows(2) {
        if pair[0].rank == pair[1].rank && pair[0].leg == pair[1].leg {
            return Err(format!("duplicate rank {} leg {}", pair[0].rank, pair[0].leg).into());
        }
        if pair[1].rank > pair[0].rank + 1 {
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

/// Resume point of a checkpointed device run. Because chain RNG streams are
/// counter-addressable (keyed by `(seed, chain, step)`), a run resumed from
/// `(cycles_done, chains)` is bit-identical to one that never stopped — the
/// checkpoint carries no RNG state at all.
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub config: DiagConfig,
    pub chain_offset: u64,
    /// Total cycles completed so far (warmup + measured).
    pub cycles_done: u64,
    /// Legs completed so far (leg k wrote `batches-rank{r}-leg{k}.bin`).
    pub leg: u64,
    /// Build provenance of the producing binary; resume refuses on drift
    /// (legs of one trajectory must run the same code) unless overridden.
    pub code: String,
    pub chains: Vec<crate::flat::FlatDiagram>,
}

impl Checkpoint {
    pub fn filename(rank: u64) -> String {
        format!("checkpoint-rank{rank}.bin")
    }

    pub fn save(&self, path: impl AsRef<Path>) -> BoxResult<()> {
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        write_header(&mut writer, CKPT_MAGIC)?;
        bincode::serialize_into(&mut writer, self)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> BoxResult<Self> {
        let file = std::fs::File::open(&path)
            .map_err(|e| format!("cannot open checkpoint '{}': {e}", path.as_ref().display()))?;
        let mut reader = std::io::BufReader::new(file);
        check_header(&mut reader, CKPT_MAGIC, "Checkpoint")?;
        Ok(bincode::deserialize_from(&mut reader)?)
    }
}
