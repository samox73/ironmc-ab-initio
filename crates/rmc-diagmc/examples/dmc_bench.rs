use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use rmc_diagmc::{run, DiagConfig, FrohlichModel};

fn main() -> rmc_core::Result<()> {
    let max_steps = std::env::args()
        .nth(1)
        .map(|value| value.parse())
        .transpose()
        .map_err(|_| rmc_core::RmcError::InvalidArgument("max_steps must be an integer".into()))?
        .unwrap_or(500_000);
    let start = Instant::now();
    let report = run(
        Arc::new(FrohlichModel::new(2.0)),
        DiagConfig {
            max_tau: 30.0,
            tau_fit: 15.0,
            max_order: 200,
            chains: 1,
            max_steps,
            warmup_steps: 0,
            fixed_tau: true,
            ..DiagConfig::default()
        },
    )?;
    println!(
        "steps/sec: {:.3}",
        max_steps as f64 / start.elapsed().as_secs_f64()
    );
    black_box(report);
    Ok(())
}
