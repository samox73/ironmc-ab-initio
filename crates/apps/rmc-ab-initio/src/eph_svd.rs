//! Per-band-pair truncated SVD of the short-range e-ph tensor.

use crate::epw::EpmatWp;
use crate::reconstruct::hermitian_eig;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::{ops::Range, path::Path};

type Cf = Complex64;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub const DEFAULT_TOL: f64 = 1e-6;
pub const MAX_RANK: usize = 100;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EphSvd {
    pub nbnd: usize,
    pub nrr_k: usize,
    pub nmodes: usize,
    pub nrr_g: usize,
    pub tol: f64,
    /// Component offsets for each `(bra,ket)` pair, plus the final total.
    pub pair_offsets: Vec<usize>,
    pub s: Vec<f64>,
    /// `[component][R_e]`.
    pub u: Vec<Cf>,
    /// `[component][R_p][cart]`; stores `U†M/σ`, so `M = U S V`.
    pub v: Vec<Cf>,
}

impl EphSvd {
    pub fn compress_default(epmat: &EpmatWp) -> Self {
        Self::compress(epmat, DEFAULT_TOL, MAX_RANK)
    }

    /// Decompose each band pair through its smaller `M M†` Gram matrix.
    pub fn compress(epmat: &EpmatWp, tol: f64, max_rank: usize) -> Self {
        assert!(tol > 0.0 && tol.is_finite());
        assert!(max_rank > 0);
        let nrow = epmat.nrr_k;
        let ncol = epmat.nrr_g * epmat.nmodes;
        assert!(
            nrow <= ncol,
            "M M† is only smaller when nrr_k <= nrr_g*nmodes"
        );

        let npair = epmat.nbnd * epmat.nbnd;
        let mut pair_offsets = Vec::with_capacity(npair + 1);
        let mut s = Vec::with_capacity(npair * max_rank);
        let mut u = Vec::with_capacity(npair * max_rank * nrow);
        let mut v = Vec::with_capacity(npair * max_rank * ncol);
        pair_offsets.push(0);

        for n in 0..epmat.nbnd {
            for m in 0..epmat.nbnd {
                let mut data = Vec::with_capacity(nrow * ncol);
                for rp in 0..epmat.nrr_g {
                    for a in 0..epmat.nmodes {
                        for re in 0..epmat.nrr_k {
                            data.push(epmat.g(re, rp, a, n, m));
                        }
                    }
                }
                let matrix = DMatrix::from_column_slice(nrow, ncol, &data);
                let gram = &matrix * matrix.adjoint();
                let mut gram_row_major = Vec::with_capacity(nrow * nrow);
                for i in 0..nrow {
                    for j in 0..nrow {
                        gram_row_major.push(gram[(i, j)]);
                    }
                }
                // Performance: benchmark faer's self-adjoint eigensolver here if
                // compression becomes a bottleneck; current faer needs a newer MSRV.
                let (evals, evecs) = hermitian_eig(&gram_row_major, nrow);
                let mut order: Vec<usize> = (0..nrow).collect();
                order.sort_by(|&a, &b| evals[b].partial_cmp(&evals[a]).unwrap());
                let sigma0 = evals[order[0]].max(0.0).sqrt();
                assert!(
                    sigma0.is_finite(),
                    "e-ph tensor has non-finite singular values"
                );
                if sigma0 > 0.0 {
                    order.retain(|&i| evals[i].max(0.0).sqrt() / sigma0 > tol);
                    if order.len() > max_rank {
                        log::debug!("SVD pair ({n},{m}) capped at {max_rank}; {} singular values exceed {tol:.1e}", order.len());
                    }
                    order.truncate(max_rank);
                    for i in order {
                        let sigma = evals[i].max(0.0).sqrt();
                        let left = DVector::from_column_slice(&evecs[i]);
                        let right = left.adjoint() * &matrix / Cf::new(sigma, 0.0);
                        s.push(sigma);
                        u.extend_from_slice(left.as_slice());
                        v.extend_from_slice(right.as_slice());
                    }
                }
                pair_offsets.push(s.len());
            }
        }

        Self {
            nbnd: epmat.nbnd,
            nrr_k: epmat.nrr_k,
            nmodes: epmat.nmodes,
            nrr_g: epmat.nrr_g,
            tol,
            pair_offsets,
            s,
            u,
            v,
        }
    }

    pub fn rank(&self) -> usize {
        self.s.len()
    }

    pub fn pair_range(&self, n: usize, m: usize) -> Range<usize> {
        let pair = n * self.nbnd + m;
        self.pair_offsets[pair]..self.pair_offsets[pair + 1]
    }

    pub fn pair_ranks(&self) -> Vec<usize> {
        self.pair_offsets.windows(2).map(|w| w[1] - w[0]).collect()
    }

    pub fn u_at(&self, component: usize, re: usize) -> Cf {
        self.u[component * self.nrr_k + re]
    }

    pub fn v_at(&self, component: usize, rp: usize, a: usize) -> Cf {
        self.v[(component * self.nrr_g + rp) * self.nmodes + a]
    }

    pub fn reconstruct(&self, re: usize, rp: usize, a: usize, n: usize, m: usize) -> Cf {
        self.pair_range(n, m)
            .map(|g| self.s[g] * self.u_at(g, re) * self.v_at(g, rp, a))
            .sum()
    }

    /// Largest element error relative to the largest input element.
    pub fn max_relative_error(&self, epmat: &EpmatWp) -> f64 {
        assert_eq!(
            (self.nbnd, self.nrr_k, self.nmodes, self.nrr_g),
            (epmat.nbnd, epmat.nrr_k, epmat.nmodes, epmat.nrr_g)
        );
        let ncol = self.nrr_g * self.nmodes;
        let scale = epmat.data.iter().map(|z| z.norm()).fold(0.0, f64::max);
        let mut worst: f64 = 0.0;
        for n in 0..self.nbnd {
            for m in 0..self.nbnd {
                let range = self.pair_range(n, m);
                let rank = range.len();
                let mut left = DMatrix::from_column_slice(
                    self.nrr_k,
                    rank,
                    &self.u[range.start * self.nrr_k..range.end * self.nrr_k],
                );
                for (column, g) in range.clone().enumerate() {
                    left.column_mut(column).scale_mut(self.s[g]);
                }
                let right = DMatrix::from_row_slice(
                    rank,
                    ncol,
                    &self.v[range.start * ncol..range.end * ncol],
                );
                let reconstructed = left * right;
                for rp in 0..self.nrr_g {
                    for a in 0..self.nmodes {
                        for re in 0..self.nrr_k {
                            let error = (reconstructed[(re, rp * self.nmodes + a)]
                                - epmat.g(re, rp, a, n, m))
                            .norm()
                                / scale;
                            worst = worst.max(error);
                        }
                    }
                }
            }
        }
        worst
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, serde_json::to_vec(self)?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gram_svd_reconstructs_a_small_tensor() {
        let epmat = EpmatWp {
            nbnd: 1,
            nrr_k: 3,
            nmodes: 1,
            nrr_g: 4,
            // 3x4, column-major, rank two.
            data: vec![
                Cf::new(1.0, 0.0),
                Cf::new(2.0, 1.0),
                Cf::new(0.0, 1.0),
                Cf::new(2.0, 0.0),
                Cf::new(4.0, 2.0),
                Cf::new(0.0, 2.0),
                Cf::new(0.0, 1.0),
                Cf::new(1.0, 0.0),
                Cf::new(1.0, 1.0),
                Cf::new(1.0, 1.0),
                Cf::new(3.0, 1.0),
                Cf::new(1.0, 2.0),
            ],
        };
        let svd = EphSvd::compress(&epmat, 1e-7, 3);
        assert_eq!(svd.pair_ranks(), vec![2]);
        assert!(svd.max_relative_error(&epmat) < 1e-10);
    }

    #[test]
    fn cache_round_trip() {
        let svd = EphSvd {
            nbnd: 1,
            nrr_k: 1,
            nmodes: 1,
            nrr_g: 1,
            tol: 1e-6,
            pair_offsets: vec![0, 1],
            s: vec![2.0],
            u: vec![Cf::new(0.0, 1.0)],
            v: vec![Cf::new(1.0, 0.0)],
        };
        let path = std::env::temp_dir().join("rmc-eph-svd-test.json");
        svd.save(&path).unwrap();
        assert_eq!(svd, EphSvd::load(&path).unwrap());
        std::fs::remove_file(path).ok();
    }
}
