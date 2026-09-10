use std::f64::consts::PI;

pub type Vec3 = [f64; 3];

pub const MASS: f64 = 1.0;
pub const OMEGA: f64 = 1.0;
pub const DELTA_TAU_LIMIT: f64 = 10.0 * f64::EPSILON;

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm_squared(p: Vec3) -> f64 {
    dot(p, p)
}

pub fn norm(p: Vec3) -> f64 {
    norm_squared(p).sqrt()
}

pub fn p0() -> f64 {
    (2.0 * MASS * OMEGA).sqrt()
}

pub fn bare_dispersion(p: Vec3) -> f64 {
    norm_squared(p) / (2.0 * MASS)
}

pub fn dispersion(p: Vec3, mu: f64) -> f64 {
    bare_dispersion(p) - mu
}

pub fn bare_propagator(p: Vec3, mu: f64, tau: f64) -> f64 {
    (-(dispersion(p, mu) * tau)).exp()
}

pub fn segment_exponent(p: Vec3, mu: f64, dtau: f64) -> f64 {
    dispersion(p, mu) * dtau
}

pub fn norm0(max_tau: f64, energy: f64) -> f64 {
    (1.0 - (-energy * max_tau).exp()) / energy
}

pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> Vec3 {
    [
        r * phi.cos() * theta.sin(),
        r * phi.sin() * theta.sin(),
        r * theta.cos(),
    ]
}

pub fn theta_from_cartesian(r: Vec3) -> f64 {
    (r[2] / norm(r)).acos()
}

pub fn phi_from_cartesian(r: Vec3) -> f64 {
    r[1].atan2(r[0])
}

/// Apply `Rz(phi) · Ry(theta)` to `v`.
///
/// GPU-portable replacement for the `nalgebra::Rotation3` composition in `ChangeQDirection`;
/// consumes no RNG and uses only `sin`/`cos`, so the same expression runs inside CubeCL kernels.
pub fn rotate_z_then_y(phi: f64, theta: f64, v: Vec3) -> Vec3 {
    let (st, ct) = (theta.sin(), theta.cos());
    let (sp, cp) = (phi.sin(), phi.cos());
    let w = [ct * v[0] + st * v[2], v[1], -st * v[0] + ct * v[2]];
    [cp * w[0] - sp * w[1], sp * w[0] + cp * w[1], w[2]]
}

pub fn draw_new_q_from_uniforms(r1: f64, r2: f64, r3: f64) -> Vec3 {
    let theta = (1.0 - 2.0 * r1).acos();
    let q = p0() / r2 - p0();
    spherical_to_cartesian(q, theta, 2.0 * PI * r3)
}

pub fn add_phonon_zero_ratio(alpha: f64, tau: f64, momentum_out: Vec3, q: Vec3) -> f64 {
    2.0 * alpha * OMEGA.powi(2) / PI
        * (-(OMEGA + (norm_squared(q) / 2.0 - dot(q, momentum_out)) / MASS) * tau).exp()
        * (1.0 + norm(q) / p0()).powi(2)
}

pub fn remove_phonon_zero_ratio(alpha: f64, tau: f64, momentum_out: Vec3, q: Vec3) -> f64 {
    PI / (2.0 * alpha * OMEGA.powi(2))
        * ((OMEGA + (norm_squared(q) / 2.0 - dot(q, momentum_out)) / MASS) * tau).exp()
        / (1.0 + norm(q) / p0()).powi(2)
}

pub fn phonon_lambda(q: Vec3) -> f64 {
    OMEGA * (1.0 + norm(q) / p0()).powi(2)
}

pub fn add_phonon_higher_ratio(
    alpha: f64,
    order: usize,
    delta_t: f64,
    q: Vec3,
    p_mean: Vec3,
    tau_span: f64,
    tail_extension_exponent: f64,
    max_tau_minus_tau1: f64,
) -> f64 {
    let algo_ratio = (2 * order - 1) as f64 / order as f64;
    algo_ratio * 2.0 * alpha * OMEGA * delta_t / PI
        * ((norm(q) * p0() + dot(q, p_mean)) * tau_span / MASS - tail_extension_exponent).exp()
        * (1.0 - (-(phonon_lambda(q) * max_tau_minus_tau1)).exp())
}

pub fn remove_phonon_higher_ratio(
    alpha: f64,
    order: usize,
    delta_t: f64,
    q: Vec3,
    p_mean: Vec3,
    tau_span: f64,
    tail_extension_exponent: f64,
    max_tau_minus_left_tau: f64,
) -> f64 {
    let algo_ratio = (order - 1) as f64 / (2 * order - 3) as f64;
    algo_ratio * PI / (2.0 * alpha * OMEGA * delta_t)
        * (-(norm(q) * p0() + dot(q, p_mean)) * tau_span / MASS + tail_extension_exponent).exp()
        / (1.0 - (-(phonon_lambda(q) * max_tau_minus_left_tau)).exp())
}

pub fn change_internal_tau_lambda(prev_p: Vec3, current_p: Vec3, incoming: bool) -> f64 {
    bare_dispersion(prev_p) - bare_dispersion(current_p) + if incoming { OMEGA } else { -OMEGA }
}

pub fn rescale_energy_term(delta_s: f64, p: Vec3, phonon_count: usize) -> f64 {
    delta_s * (bare_dispersion(p) + OMEGA * phonon_count as f64)
}

pub fn rescale_diagram_ratio(order: usize, tau: f64, tau_prime: f64, energy: f64) -> f64 {
    let n = (order - 1) as f64;
    (2.0 * n * (tau_prime / tau).ln() - energy * (tau_prime - tau)
        + ((energy * tau_prime - 2.0 * n).powi(2) - (energy * tau - 2.0 * n).powi(2)) / (4.0 * n))
        .exp()
}

pub fn change_q_modulus_sigma(tau_span: f64) -> f64 {
    (MASS / tau_span).sqrt()
}

pub fn change_q_direction_a(tau_span: f64, p_mean_norm: f64, q_norm: f64) -> f64 {
    tau_span * p_mean_norm * q_norm / MASS
}

pub fn change_topology_ratio(dtau: f64, p_prime: Vec3, p_old: Vec3, c1: f64, c2: f64) -> f64 {
    (-(dtau) * (bare_dispersion(p_prime) - bare_dispersion(p_old) - OMEGA * (c1 - c2))).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_geometry_round_trips() {
        let v = spherical_to_cartesian(2.0, 0.7, 1.2);
        assert!((norm(v) - 2.0).abs() < 1.0e-12);
        assert!((theta_from_cartesian(v) - 0.7).abs() < 1.0e-12);
        assert!((phi_from_cartesian(v) - 1.2).abs() < 1.0e-12);
    }

    #[test]
    fn rotate_z_then_y_matches_nalgebra() {
        use nalgebra::{Rotation3, Vector3};
        for (phi, theta, v) in [
            (0.3, 1.1, [0.25, -0.5, 1.25]),
            (-2.4, 0.0, [1.0, 0.0, 0.0]),
            (0.0, 3.0, [0.0, -1.0, 2.0]),
        ] {
            let expected = (Rotation3::from_axis_angle(&Vector3::z_axis(), phi)
                * Rotation3::from_axis_angle(&Vector3::y_axis(), theta))
                * Vector3::new(v[0], v[1], v[2]);
            let actual = rotate_z_then_y(phi, theta, v);
            for i in 0..3 {
                assert!(
                    (actual[i] - expected[i]).abs() < 1.0e-14,
                    "component {i}: {} vs {}",
                    actual[i],
                    expected[i]
                );
            }
        }
    }
}
