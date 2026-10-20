//! Ab-initio DiagMC prototype (LiF).
//!
//! Phase A (`epb`/`epw`/`longrange`/`model`/`reconstruct`): EPW tensor readers +
//! the reduced `|g(Γ,q,LO)|²` artifact. Phase B (`vertex`/`updates`/`config`/
//! `run`): the DiagMC consumer — reuse the Fröhlich engine and swap its vertex by
//! reweighting the acceptance. See `plan-rmc-ab-initio.md`. patt126's `IO.py` is a
//! reference only.

pub mod config;
pub mod dump;
pub mod epb;
pub mod eph_svd;
pub mod epw;
pub mod flat;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod longrange;
pub mod model;
pub mod model_abinitio;
pub mod reconstruct;
pub mod run;
pub mod updates;
pub mod vertex;
pub mod vertex_direct;
pub mod vertex_table;

/// Git provenance of this build (`<short-hash>` or `<short-hash>-dirty`),
/// embedded at compile time; recorded in dumps, checkpoints, and reports.
pub fn code_version() -> &'static str {
    env!("IRONMC_GIT_HASH")
}
