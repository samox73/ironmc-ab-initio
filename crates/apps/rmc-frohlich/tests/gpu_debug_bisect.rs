//! Debug harness: full-state GPU-vs-CPU comparison over the first steps. When `gpu_parity`
//! fails, this localizes the first divergent step/chain/update and inverts the ChangeTau
//! sampler to recover the uniform draw each side used.
#![cfg(any(feature = "gpu-cpu", feature = "gpu-hip", feature = "gpu-cuda"))]

use cubecl::prelude::Runtime;
use rmc_frohlich::config::RunConfig;
use rmc_frohlich::flat::philox::keyed_draw;
use rmc_frohlich::gpu::kernel::launch_reference_kernel;
use rmc_frohlich::gpu::run::{run_device, SelectedRuntime};

#[test]
fn find_first_divergent_step() {
    let base_cfg = RunConfig {
        alpha: 2.0,
        chains: 130,
        steps_per_cycle: 1,
        n_batches: 4,
        num_bins: 50,
        max_order_gpu: 64,
        initial_self_consistent_period: usize::MAX,
        seed: 0xB0BA_CAFE_D00D_1234,
        ..RunConfig::default()
    };
    let n_chains = base_cfg.chains as usize;

    for steps in 1..=8u64 {
        let mut cfg = base_cfg.clone();
        cfg.max_steps = steps;
        let cpu = launch_reference_kernel(&cfg).unwrap();
        let client = <SelectedRuntime as Runtime>::client(&Default::default());
        let gpu = run_device::<SelectedRuntime>(&cfg, client).unwrap();

        for chain in 0..n_chains {
            let c = &cpu.chains[chain];
            let g = &gpu.chains[chain];
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
            let mut bad = Vec::new();
            if g.order != c.order {
                bad.push(format!("order {} vs {}", g.order, c.order));
            }
            for slot in 0..c.capacity() {
                if !close(g.tau[slot], c.tau[slot]) {
                    bad.push(format!("tau[{slot}] {} vs {}", g.tau[slot], c.tau[slot]));
                }
                for k in 0..3 {
                    if !close(g.p_out[slot][k], c.p_out[slot][k]) {
                        bad.push(format!(
                            "p_out[{slot}][{k}] {} vs {}",
                            g.p_out[slot][k], c.p_out[slot][k]
                        ));
                    }
                    if !close(g.q[slot][k], c.q[slot][k]) {
                        bad.push(format!("q[{slot}][{k}] {} vs {}", g.q[slot][k], c.q[slot][k]));
                    }
                }
            }
            if g.link != c.link {
                bad.push(format!("link {:?} vs {:?}", g.link, c.link));
            }
            if !bad.is_empty() {
                let group = chain / 64;
                let step = steps - 1;
                let upd = keyed_draw(cfg.seed, group as u64, step, u32::MAX)[0] % 8;

                // Recover the uniform draw each side used for ChangeTau by inverting
                // exponential_sample_bounded on the (identical) pre-step state.
                let mut prev_cfg = cfg.clone();
                prev_cfg.max_steps = steps - 1;
                let prev = launch_reference_kernel(&prev_cfg).unwrap();
                let d = &prev.chains[chain];
                let sl = d.prev[d.tail as usize] as usize;
                let p = d.p_out[sl];
                let lambda = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) / 2.0 - d.mu
                    + if d.order != 0 { 1.0 } else { 0.0 };
                let a = d.tau[sl];
                let b = d.max_tau;
                let invert = |tau_p: f64| {
                    (-(lambda * (tau_p - a))).exp_m1() / (-(lambda * (b - a))).exp_m1()
                };
                let word = |w: u32| {
                    keyed_draw(cfg.seed, chain as u64, step, w / 4)[w as usize % 4]
                };
                let u01_lohi = |w0: u32, w1: u32| {
                    let bits = u64::from(word(w0)) | (u64::from(word(w1)) << 32);
                    (bits >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
                };
                panic!(
                    "after {steps} steps, chain {chain} (group {group}, last update {upd}) \
                     differs (gpu vs cpu):\n{}\nlambda {lambda}\n\
                     r inverted from cpu tau: {}\nr inverted from gpu tau: {}\n\
                     u01 words(0,1) lo-first: {}\nu01 words(0,1) hi-first: {}\n\
                     u01 words(1,2): {}\nu01 words(2,3): {}\nu01 words(4,5): {}",
                    bad.join("\n"),
                    invert(cpu.chains[chain].tau[cpu.chains[chain].tail as usize]),
                    invert(gpu.chains[chain].tau[gpu.chains[chain].tail as usize]),
                    u01_lohi(0, 1),
                    {
                        let bits = u64::from(word(1)) | (u64::from(word(0)) << 32);
                        (bits >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
                    },
                    u01_lohi(1, 2),
                    u01_lohi(2, 3),
                    u01_lohi(4, 5),
                );
            }
        }
        println!("steps {steps}: all chains match");
    }
}
