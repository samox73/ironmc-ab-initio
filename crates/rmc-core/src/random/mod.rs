//! Random-number helpers and deterministic per-chain seeding.
//!
//! The framework uses [`SeedSource`] to derive one reproducible RNG stream per independent chain.
//! Distribution helpers in this module are deliberately small analytical building blocks used by
//! tests and examples; they are not intended to replace the broader `rand_distr` ecosystem.

mod samples;
mod seed_source;

pub use rand::Rng;
pub use samples::{
    cauchy_pdf, cauchy_sample, exclusive_uniform_int_pdf, exponential_pdf, exponential_pdf_bounded,
    exponential_sample, exponential_sample_bounded, normal_from_uniforms, normal_pdf,
    safe_exponential_pdf, safe_exponential_sample, uniform_index, uniform_index_from_u01,
    uniform_int_pdf, uniform_pdf, uniform_sample,
};
pub use seed_source::{ChainId, DefaultRng, SeedSource};
