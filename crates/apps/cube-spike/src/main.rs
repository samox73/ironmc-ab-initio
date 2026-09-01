#[cfg(any(
    feature = "cubecl-cpu",
    feature = "cubecl-hip",
    feature = "cubecl-cuda"
))]
fn run<R: cubecl::prelude::Runtime>(client: cubecl::prelude::ComputeClient<R>) {
    env_logger::init();
    let dtau = [0.0, 0.1, 1.0, 12.5];
    let exp = cube_spike::cube::run_exp::<R>(client.clone(), 0.7, &dtau);
    let expected = cube_spike::exp_reference(0.7, &dtau);
    assert!(exp
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() <= expected.abs().max(1.0) * 1e-14));

    let trait_out = cube_spike::cube::run_trait::<R>(client.clone(), &dtau);
    assert_eq!(trait_out.len(), dtau.len());

    let loop_out = cube_spike::cube::run_loop_branch::<R>(client.clone(), &[0, 10, 20], 4);
    assert_eq!(
        loop_out,
        cube_spike::divergent_loop_reference(&[0, 10, 20], 4)
    );

    let atomic = cube_spike::cube::run_atomic_f64_add::<R>(client.clone());
    let philox = cube_spike::cube::run_philox::<R>(client.clone(), [0, 0, 0, 0], [0, 0]);
    assert_eq!(
        philox,
        cube_spike::philox::philox4x32_10([0, 0, 0, 0], [0, 0])
    );

    let words = cube_spike::cube::run_philox_words::<R>(client, 1_000_000);
    for (block, actual) in words.chunks_exact(4).enumerate() {
        let expected = cube_spike::philox::philox4x32_10([block as u32, 0, 0, 0], [0, 0]);
        assert_eq!(actual, expected);
    }

    println!("exp: pass");
    println!("trait_composition: pass");
    println!("loop_branch: pass");
    println!("atomic_f64_add: {atomic:?}");
    println!("philox: pass (1_000_000 words)");
}

#[cfg(feature = "cubecl-cpu")]
fn main() {
    let client = <cubecl::cpu::CpuRuntime as cubecl::prelude::Runtime>::client(&Default::default());
    run::<cubecl::cpu::CpuRuntime>(client);
}

#[cfg(all(not(feature = "cubecl-cpu"), feature = "cubecl-hip"))]
fn main() {
    let client = <cubecl::hip::HipRuntime as cubecl::prelude::Runtime>::client(&Default::default());
    run::<cubecl::hip::HipRuntime>(client);
}

#[cfg(all(
    not(feature = "cubecl-cpu"),
    not(feature = "cubecl-hip"),
    feature = "cubecl-cuda"
))]
fn main() {
    let client =
        <cubecl::cuda::CudaRuntime as cubecl::prelude::Runtime>::client(&Default::default());
    run::<cubecl::cuda::CudaRuntime>(client);
}

#[cfg(not(any(
    feature = "cubecl-cpu",
    feature = "cubecl-hip",
    feature = "cubecl-cuda"
)))]
fn main() {
    println!("Build with --features cubecl-cpu, cubecl-hip, or cubecl-cuda to run CubeCL kernels.");
}
