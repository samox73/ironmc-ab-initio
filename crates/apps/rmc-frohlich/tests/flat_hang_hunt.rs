//! Debug harness for the 8192-chain hang: replay chains independently, find the first chain that
//! panics, then replay it with per-step invariant checks to locate the first corruption.

use rand::Rng;
use rmc_core::mc::{Update, WeightedUpdateSet};
use rmc_frohlich::config::RunConfig;
use rmc_frohlich::flat::philox::{keyed_draw, PhiloxRng};
use rmc_frohlich::flat::updates::{default_update_set, FlatPolaronUpdate};
use rmc_frohlich::flat::{FlatDiagram, NULL};

const GROUP_SIZE: u64 = 64;

fn hang_cfg() -> RunConfig {
    RunConfig {
        alpha: 2.0,
        chains: 8192,
        max_steps: 50_000,
        steps_per_cycle: 100,
        warmup_steps: 0,
        n_batches: 8,
        num_bins: 100,
        ..RunConfig::default()
    }
}

fn build_chain(cfg: &RunConfig) -> FlatDiagram {
    FlatDiagram::with_parameters(
        cfg.alpha,
        cfg.mu,
        cfg.momentum,
        cfg.max_tau,
        cfg.start_tau,
        cfg.min_order,
        cfg.max_order,
        cfg.max_order_gpu,
    )
}

fn step(
    cfg: &RunConfig,
    chain_id: u64,
    step: u64,
    d: &mut FlatDiagram,
    updates: &mut WeightedUpdateSet<FlatPolaronUpdate>,
) -> usize {
    let group = chain_id / GROUP_SIZE;
    let upd = (keyed_draw(cfg.seed, group, step, u32::MAX)[0] as usize) % 8;
    let mut rng = PhiloxRng::new(cfg.seed, chain_id, step);
    let entry = &mut updates.entries_mut()[upd];
    let probability = entry.update_mut().attempt(d, &mut rng) * entry.ratio();
    if probability < 0.0 {
        entry.update_mut().reject(d);
    } else if probability >= 1.0 || rng.gen::<f64>() < probability {
        entry.update_mut().accept(d);
    } else {
        entry.update_mut().reject(d);
    }
    upd
}

/// Structural invariants: strictly increasing taus along the list, symmetric links, list
/// reachability consistent with storage size.
fn check_invariants(d: &FlatDiagram) -> Result<(), String> {
    let mut slot = d.head;
    let mut count = 0usize;
    let mut last_tau = f64::NEG_INFINITY;
    while slot != NULL {
        let s = slot as usize;
        if d.tau[s] <= last_tau {
            return Err(format!(
                "tau not strictly increasing at slot {slot}: {} <= {}",
                d.tau[s], last_tau
            ));
        }
        last_tau = d.tau[s];
        let link = d.link[s];
        if link != NULL && d.link[link as usize] != slot {
            return Err(format!("asymmetric link at slot {slot}"));
        }
        count += 1;
        if count > d.capacity() {
            return Err("list cycle detected".to_string());
        }
        slot = d.next[s];
    }
    if count != d.vertex_count() {
        return Err(format!(
            "list length {count} != vertex count {}",
            d.vertex_count()
        ));
    }
    Ok(())
}

#[test]
#[ignore = "debug harness for the 8192-chain hang; run explicitly"]
fn hunt_first_corrupt_chain() {
    let cfg = hang_cfg();

    // Phase 1: replay chains independently, silently catching panics.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut failing = None;
    for chain_id in 0..cfg.chains {
        let result = std::panic::catch_unwind(|| {
            let mut d = build_chain(&cfg);
            let mut updates = default_update_set().unwrap();
            for s in 0..cfg.max_steps {
                step(&cfg, chain_id, s, &mut d, &mut updates);
            }
        });
        if result.is_err() {
            failing = Some(chain_id);
            break;
        }
    }
    std::panic::set_hook(default_hook);
    let Some(chain_id) = failing else {
        println!("all {} chains replayed clean", cfg.chains);
        return;
    };
    println!("first panicking chain: {chain_id}");

    // Phase 2: replay that chain with invariant checks each step; report first violation.
    let mut d = build_chain(&cfg);
    let mut updates = default_update_set().unwrap();
    let mut prev_state = d.clone();
    for s in 0..cfg.max_steps {
        let upd = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            step(&cfg, chain_id, s, &mut d, &mut updates)
        }));
        match upd {
            Err(_) => {
                println!("chain {chain_id} PANICKED during step {s}");
                println!(
                    "state before step (order {}, vertices {}):\n{}",
                    prev_state.order,
                    prev_state.vertex_count(),
                    serde_json::to_string(&prev_state).unwrap()
                );
                panic!("panic located at step {s}");
            }
            Ok(upd) => {
                if let Err(msg) = check_invariants(&d) {
                    println!("chain {chain_id} INVARIANT VIOLATION after step {s} (update {upd}): {msg}");
                    println!(
                        "state before step:\n{}",
                        serde_json::to_string(&prev_state).unwrap()
                    );
                    println!("state after step:\n{}", serde_json::to_string(&d).unwrap());
                    panic!("invariant violated at step {s}");
                }
            }
        }
        prev_state = d.clone();
    }
    panic!("chain {chain_id} replayed clean solo — corruption requires lockstep context?");
}
