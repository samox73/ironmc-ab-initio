//! Feature-gated GPU entry points.
//!
//! With a runtime feature (`gpu-cpu`, `gpu-hip`, `gpu-cuda`) the `gpu` subcommand runs the real
//! CubeCL kernel (`kernel::mc_kernel`) on that runtime; the CubeCL CPU runtime doubles as the
//! locally runnable correctness target. With only the base `gpu` feature the Phase-2 batched CPU
//! rig (`kernel::launch_reference_kernel`) is used — it remains the parity oracle in tests.

pub mod kernel;
pub mod physics;
pub mod run;
pub mod state;

pub use run::run_gpu_from_config;
