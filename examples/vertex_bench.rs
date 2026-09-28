use std::hint::black_box;
use std::time::Instant;

use num_complex::Complex64;
use rmc_ab_initio::vertex_table::{Tab, VertexTable, DEFAULT_PHFREQ_CUTOFF_EV};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let calls = std::env::args()
        .nth(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(2_000_000_usize);
    let table = synthetic_table()?;
    let len = table.len();

    let start = Instant::now();
    let mut sum = Complex64::new(0.0, 0.0);
    for i in 0..calls {
        sum += table.short_indices(i * 17 % len, (i * 31 + 1) % len, i % table.nmodes);
    }
    let elapsed = start.elapsed();
    black_box(sum);
    println!("calls/sec: {:.3}", calls as f64 / elapsed.as_secs_f64());
    Ok(())
}

fn synthetic_table() -> Result<VertexTable, Box<dyn std::error::Error>> {
    let grid_n = 8_usize;
    let nbnd = 4_usize;
    let nmodes = 6_usize;
    let rank_per_pair = 10_usize;
    let nk = grid_n.pow(3);
    let components = nbnd.pow(2) * rank_per_pair;
    let complex = |i: usize| {
        Complex64::new(
            ((i * 17 % 101) as f64 + 1.0) / 101.0,
            ((i * 29 % 103) as f64 + 1.0) / 103.0,
        )
    };
    Ok(VertexTable {
        grid_n,
        nbnd,
        nmodes,
        pair_offsets: (0..=nbnd.pow(2)).map(|pair| pair * rank_per_pair).collect(),
        s: (0..components)
            .map(|i| (i % 13 + 1) as f64 / 13.0)
            .collect(),
        wtab: Tab::F64((0..nk * components).map(complex).collect()),
        bra_band: (0..components)
            .map(|g| (g / rank_per_pair / nbnd) as u8)
            .collect(),
        vtab: Tab::F64((0..nk * nmodes * components).map(complex).collect()),
        cbm_evec: (0..nk * nbnd).map(complex).collect(),
        electron_ev: vec![0.0; nk],
        phonon_ev: vec![1.0; nk * nmodes],
        long_ev: vec![Complex64::new(0.0, 0.0); nk * nmodes],
        phfreq_cutoff_ev: DEFAULT_PHFREQ_CUTOFF_EV,
    })
}
