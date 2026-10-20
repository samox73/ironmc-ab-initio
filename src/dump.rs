//! Per-rank batch dumps + offline merge (the `multi-node` design, consumed by
//! CPU ranks and GPU ranks alike). A dump carries the raw `MeasurementData`
//! plus the one model scalar `summarize` needs, so merging never reloads a
//! vertex table. Merging sorts by rank and concatenates batch lists in that
//! fixed order — the merged report is a pure function of (seed, total chains).

use std::path::Path;

use rmc_core::random::DefaultRng;
use rmc_core::Merge;
use rmc_diagmc::measure::{summarize, MeasurementData};
use rmc_diagmc::{DiagConfig, Diagram, Report};
use serde::{Deserialize, Serialize};

type BoxResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// Versioned file headers (the VertexTable::save idiom): adding fields to
// these bincode payloads is a version bump, not silent data orphaning.
const DUMP_MAGIC: &[u8; 8] = b"RMCDUMP\0";
const CKPT_MAGIC: &[u8; 8] = b"RMCCKPT\0";
const SLOT_CKPT_MAGIC: &[u8; 8] = b"RMCSCKP\0";
// TODO: at the next FORMAT_VERSION bump, move the flat engines' `group_size`
// into `DiagConfig` (serde default 1; the slotmap engine ignores it) so
// `same_trajectory` and checkpoints catch group drift across legs — free once
// the on-disk payloads are being regenerated anyway (see
// `BatchedRunOpts::group_size` for why it is not there today).
const FORMAT_VERSION: u32 = 1;

fn write_header(writer: &mut impl std::io::Write, magic: &[u8; 8]) -> BoxResult<()> {
    writer.write_all(magic)?;
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
    Ok(())
}

/// Header + payload to `<path>.tmp`, fsync, rename over `path`: a kill
/// mid-write (OOM, node failure — the very cases checkpoints exist for) must
/// never truncate the previous good file, which for a checkpoint is the run's
/// only resume point.
fn save_atomic(path: &Path, magic: &[u8; 8], value: &impl Serialize) -> BoxResult<()> {
    let tmp = path.with_extension("bin.tmp");
    let mut writer = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    write_header(&mut writer, magic)?;
    bincode::serialize_into(&mut writer, value)?;
    writer
        .into_inner()
        .map_err(std::io::IntoInnerError::into_error)?
        .sync_all()?;
    std::fs::rename(&tmp, path)?;
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
        save_atomic(path.as_ref(), DUMP_MAGIC, self)
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
///
/// Blind spot: the flat engines' `group_size` also shapes the trajectory but
/// lives outside `DiagConfig` (see `BatchedRunOpts::group_size` for why), so
/// config equality cannot see it — `run-flat` legs must repeat the same
/// `--group` themselves.
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
            return Err(
                format!("rank {} config differs from rank {}", dump.rank, first.rank).into(),
            );
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
    /// Build provenance of the producing binary — recorded so a run's results
    /// can be reproduced by rebuilding at this hash (resume warns on drift).
    pub code: String,
    pub chains: Vec<crate::flat::FlatDiagram>,
}

impl Checkpoint {
    pub fn filename(rank: u64) -> String {
        format!("checkpoint-rank{rank}.bin")
    }

    pub fn save(&self, path: impl AsRef<Path>) -> BoxResult<()> {
        save_atomic(path.as_ref(), CKPT_MAGIC, self)
    }

    pub fn load(path: impl AsRef<Path>) -> BoxResult<Self> {
        let file = std::fs::File::open(&path)
            .map_err(|e| format!("cannot open checkpoint '{}': {e}", path.as_ref().display()))?;
        let mut reader = std::io::BufReader::new(file);
        check_header(&mut reader, CKPT_MAGIC, "Checkpoint")?;
        Ok(bincode::deserialize_from(&mut reader)?)
    }
}

/// Resume point of a checkpointed *slotmap* run — the CPU-fast engine behind
/// `run-cpu`. Unlike the device engine's counter-addressable Philox RNG, the
/// slotmap engine's `Xoshiro256PlusPlus` is stateful, so the checkpoint stores
/// each chain's `(Diagram, rng)` in full; restoring both continues the exact
/// trajectory. Chains are async (each stops at its own step when the deadline
/// hits), so there is no single `cycles_done` — legs accumulate via `merge`.
///
/// Bit-identity caveat: the drift-check resync (exact weight recompute every
/// 4096 cycles) counts from *leg start*, so legs long enough for it to fire
/// resync at different absolute steps than an uninterrupted run — the cached
/// weight's low bits can then differ and later accept decisions may flip.
/// Estimates stay exact either way; only bit-level replay of a *split* run is
/// conditional on no resync firing.
#[derive(Serialize, Deserialize)]
pub struct SlotmapCheckpoint {
    pub config: DiagConfig,
    pub chain_offset: u64,
    /// Legs completed so far (leg k wrote `batches-rank{r}-leg{k}.bin`).
    pub leg: u64,
    /// Build provenance of the producing binary — recorded so a run's results
    /// can be reproduced by rebuilding at this hash (resume warns on drift).
    pub code: String,
    pub chains: Vec<(Diagram, DefaultRng)>,
}

impl SlotmapCheckpoint {
    pub fn filename(rank: u64) -> String {
        format!("checkpoint-rank{rank}.bin")
    }

    pub fn save(&self, path: impl AsRef<Path>) -> BoxResult<()> {
        save_atomic(path.as_ref(), SLOT_CKPT_MAGIC, self)
    }

    pub fn load(path: impl AsRef<Path>) -> BoxResult<Self> {
        let file = std::fs::File::open(&path)
            .map_err(|e| format!("cannot open checkpoint '{}': {e}", path.as_ref().display()))?;
        let mut reader = std::io::BufReader::new(file);
        check_header(&mut reader, SLOT_CKPT_MAGIC, "SlotmapCheckpoint")?;
        Ok(bincode::deserialize_from(&mut reader)?)
    }
}
