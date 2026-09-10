//! Feature-gated GPU code for the ab-initio engine (plan `gpu-abinitio`).
//!
//! Phase A1: the vertex-dot microbenchmark (`vertex_bench`) — the GO/NO-GO
//! bandwidth measurement on `g(k,q,ν)` evaluation, validated against
//! `VertexTable::short_indices_capped`. Runtime selection mirrors
//! `rmc_frohlich::gpu::run`: CUDA > HIP > CPU by feature precedence.

pub mod kernel;
pub mod run;
pub mod vertex_bench;

#[cfg(feature = "gpu-cuda")]
pub type SelectedRuntime = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
pub type SelectedRuntime = cubecl::hip::HipRuntime;
#[cfg(all(
    feature = "gpu-cpu",
    not(feature = "gpu-hip"),
    not(feature = "gpu-cuda")
))]
pub type SelectedRuntime = cubecl::cpu::CpuRuntime;
