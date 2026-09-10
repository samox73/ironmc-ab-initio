use std::f64::consts::PI;

use rand::Rng;

/// Draw a uniform index in `0..len` without modulo bias.
///
/// Prefer this over `rng.next_u64() as usize % len`: the modulo pattern skews the distribution when
/// `len` does not divide `2^64`. Delegates to `rand`'s `gen_range`, which uses rejection sampling.
///
/// # Panics
/// Panics if `len == 0`.
pub fn uniform_index<R: Rng + ?Sized>(rng: &mut R, len: usize) -> usize {
    assert!(len > 0, "uniform_index requires len > 0");
    rng.gen_range(0..len)
}

pub const fn uniform_sample(r: f64, a: f64, b: f64) -> f64 {
    assert!(b > a);
    a + r * (b - a)
}

pub const fn uniform_pdf(a: f64, b: f64) -> f64 {
    assert!(b > a);
    1.0 / (b - a)
}

pub fn exponential_sample(r: f64, lambda: f64, a: f64) -> f64 {
    assert!(lambda > 0.0);
    a - r.ln() / lambda
}

pub fn exponential_pdf(x: f64, lambda: f64, a: f64) -> f64 {
    assert!(lambda > 0.0);
    lambda * (-(lambda * (x - a))).exp()
}

pub fn exponential_sample_bounded(r: f64, lambda: f64, a: f64, b: f64) -> f64 {
    assert!(lambda != 0.0);
    assert!(b > a);
    // exp_m1/ln_1p keep full precision when lambda*(b-a) is small.
    a - (r * (-(lambda * (b - a))).exp_m1()).ln_1p() / lambda
}

pub fn exponential_pdf_bounded(x: f64, lambda: f64, a: f64, b: f64) -> f64 {
    assert!(lambda != 0.0);
    assert!(b > a);
    lambda / -(-(lambda * (b - a))).exp_m1() * (-(lambda * (x - a))).exp()
}

pub fn safe_exponential_sample(r: f64, lambda: f64, a: f64, b: f64) -> f64 {
    if lambda > 0.0 {
        exponential_sample_bounded(r, lambda, a, b)
    } else if lambda < 0.0 {
        b - exponential_sample_bounded(r, -lambda, 0.0, b - a)
    } else {
        uniform_sample(r, a, b)
    }
}

pub fn safe_exponential_pdf(x: f64, lambda: f64, a: f64, b: f64) -> f64 {
    if lambda > 0.0 {
        exponential_pdf_bounded(x, lambda, a, b)
    } else if lambda < 0.0 {
        exponential_pdf_bounded(b - x, -lambda, 0.0, b - a)
    } else {
        uniform_pdf(a, b)
    }
}

/// Uniform index in `0..len` from a single uniform draw `r ∈ [0, 1)`.
///
/// GPU-portable replacement for [`uniform_index`]: exactly one uniform is consumed, so CPU and
/// device streams stay word-aligned. The floor map carries a relative bias of order `len / 2^53`,
/// negligible against MC statistics for any realistic vertex count.
///
/// # Panics
/// Panics if `len == 0`.
pub fn uniform_index_from_u01(r: f64, len: usize) -> usize {
    assert!(len > 0, "uniform_index_from_u01 requires len > 0");
    ((r * len as f64) as usize).min(len - 1)
}

/// Normal sample from two uniform draws `r1, r2 ∈ [0, 1)` via Box-Muller,
/// `mean + sigma * sqrt(-2 ln(1 - r1)) * cos(2π r2)`.
///
/// GPU-portable replacement for `rand_distr::Normal` (ziggurat draws a data-dependent number of
/// words); this consumes exactly two uniforms. `ln_1p(-r1)` keeps the log argument strictly
/// positive for every representable `r1 < 1`.
pub fn normal_from_uniforms(r1: f64, r2: f64, mean: f64, sigma: f64) -> f64 {
    let radius = (-2.0 * (-r1).ln_1p()).sqrt();
    mean + sigma * radius * (2.0 * PI * r2).cos()
}

pub fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    assert!(sigma > 0.0);
    let tmp = (x - mu) / sigma;
    (-0.5 * tmp * tmp).exp() / (sigma * (2.0 * PI).sqrt())
}

pub fn cauchy_sample(r: f64, x0: f64, gamma: f64) -> f64 {
    assert!(gamma > 0.0);
    x0 + gamma * (PI * (r - 0.5)).tan()
}

pub fn cauchy_pdf(x: f64, x0: f64, gamma: f64) -> f64 {
    assert!(gamma > 0.0);
    1.0 / (PI * gamma * (1.0 + ((x - x0) / gamma).powi(2)))
}

pub fn uniform_int_pdf<T>(a: T, b: T) -> f64
where
    T: Into<i128> + Copy,
{
    let a = a.into();
    let b = b.into();
    assert!(b >= a);
    1.0 / ((b - a + 1) as f64)
}

pub fn exclusive_uniform_int_pdf<T>(a: T, b: T) -> f64
where
    T: Into<i128> + Copy,
{
    let a = a.into();
    let b = b.into();
    assert!(b > a);
    1.0 / ((b - a) as f64)
}
