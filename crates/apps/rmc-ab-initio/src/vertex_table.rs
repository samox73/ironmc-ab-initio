//! Uniform-grid Fourier tables for the SVD-compressed vertex.

use crate::eph_svd::EphSvd;
use crate::epw::{Crystal, EpwData, Wigner};
use crate::longrange::{g_long_cart, project_cartesian_branches, quantized_eigenvectors};
use crate::reconstruct::{interpolate_d_polar, interpolate_h, phase, RY_TO_EV, RY_TO_THZ};
use crate::vertex_direct::DirectVertex;
use nalgebra::Vector3;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

type Cf = Complex64;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub const DEFAULT_PHFREQ_CUTOFF_EV: f64 = 1.0e-3;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VertexTable {
    pub grid_n: usize,
    pub nbnd: usize,
    pub nmodes: usize,
    pub pair_offsets: Vec<usize>,
    pub s: Vec<f64>,
    /// `[k][component]`, precontracted with the singular value and CBM ket.
    #[serde(alias = "utab")]
    pub wtab: Vec<Cf>,
    /// CBM bra-band index for each compressed component.
    #[serde(default)]
    pub bra_band: Vec<u8>,
    /// `[component][q][branch]`.
    pub vtab: Vec<Cf>,
    /// CBM eigenvector `[k][orbital]`.
    pub cbm_evec: Vec<Cf>,
    /// CBM dispersion in eV, shifted so the Γ band edge is zero.
    pub electron_ev: Vec<f64>,
    /// Physical phonon energies `[q][branch]` in eV.
    pub phonon_ev: Vec<f64>,
    /// Polar Cartesian vertex projected onto phonon branches, `[q][branch]` in eV.
    pub long_ev: Vec<Cf>,
    #[serde(default = "default_phfreq_cutoff_ev")]
    pub phfreq_cutoff_ev: f64,
}

fn default_phfreq_cutoff_ev() -> f64 {
    DEFAULT_PHFREQ_CUTOFF_EV
}

impl VertexTable {
    pub fn estimated_bytes(grid_n: usize, components: usize, nbnd: usize, nmodes: usize) -> u64 {
        let nk = grid_n as u64 * grid_n as u64 * grid_n as u64;
        let complex =
            nk * (components as u64 * (1 + nmodes) as u64 + nbnd as u64 + nmodes as u64) * 16;
        let real = nk * (1 + nmodes) as u64 * 8 + components as u64 * 8;
        complex + real
    }

    pub fn build(
        epw: &EpwData,
        crystal: &Crystal,
        wigner: &Wigner,
        svd: &EphSvd,
        grid_n: usize,
        phfreq_cutoff_ev: f64,
    ) -> Result<Self> {
        if grid_n == 0 {
            return Err("grid_n must be positive".into());
        }
        if !phfreq_cutoff_ev.is_finite() || phfreq_cutoff_ev < 0.0 {
            return Err("phfreq_cutoff_ev must be finite and non-negative".into());
        }
        if (svd.nbnd, svd.nrr_k, svd.nmodes, svd.nrr_g)
            != (epw.nbnd, epw.nrr_k, epw.nmodes, epw.nrr_g)
        {
            return Err("SVD and EPW dimensions differ".into());
        }
        if epw.nbnd > u8::MAX as usize + 1 {
            return Err("precontracted vertex supports at most 256 bands".into());
        }
        let bytes = Self::estimated_bytes(grid_n, svd.rank(), epw.nbnd, epw.nmodes);
        log::debug!(
            "vertex table estimate: {:.2} GiB",
            bytes as f64 / 2_f64.powi(30)
        );
        if bytes > 3 * 1024_u64.pow(3) {
            return Err(format!(
                "USER CHECKPOINT: vertex tables need {:.2} GiB",
                bytes as f64 / 2_f64.powi(30)
            )
            .into());
        }

        let nk = grid_n.pow(3);
        let nb = epw.nbnd;
        let nm = epw.nmodes;
        let components = svd.rank();
        let mut electron_ev = vec![0.0; nk];
        let mut cbm_evec = vec![Cf::new(0.0, 0.0); nk * nb];
        let mut phonon_ev = vec![0.0; nk * nm];
        let mut phonon_evec = vec![Cf::new(0.0, 0.0); nk * nm * nm];
        let mut long_ev = vec![Cf::new(0.0, 0.0); nk * nm];

        for index in 0..nk {
            let point = Self::point_for(grid_n, index);
            let (energies, evecs) = interpolate_h(epw, wigner, point);
            electron_ev[index] = energies[0];
            cbm_evec[index * nb..(index + 1) * nb].copy_from_slice(&evecs[0]);

            let phonons = interpolate_d_polar(epw, crystal, wigner, point);
            let gl = g_long_cart(crystal, epw, point);
            let branches = project_cartesian_branches(crystal, &phonons, &gl, phfreq_cutoff_ev);
            let evec = quantized_eigenvectors(crystal, &phonons, phfreq_cutoff_ev);
            for nu in 0..nm {
                let omega_ry = phonons.omega_thz[nu].abs() / RY_TO_THZ;
                phonon_ev[index * nm + nu] = omega_ry * RY_TO_EV;
                for a in 0..nm {
                    let e = evec[nu * nm + a];
                    phonon_evec[(index * nm + nu) * nm + a] = e;
                }
                long_ev[index * nm + nu] = branches[nu];
            }
        }
        // Shift to the global band minimum so ε ≥ 0 holds for any material;
        // for a Γ-minimum band (LiF) this equals electron_ev[0].
        let band_edge = electron_ev.iter().copied().fold(f64::INFINITY, f64::min);
        if band_edge < electron_ev[0] {
            log::warn!(
                "conduction minimum is {:.6} eV below Γ — the band edge is off-Γ; \
                 dispersion is shifted to the true minimum",
                electron_ev[0] - band_edge
            );
        }
        electron_ev.iter_mut().for_each(|e| *e -= band_edge);

        let mut wtab = vec![Cf::new(0.0, 0.0); components * nk];
        let mut vtab = vec![Cf::new(0.0, 0.0); components * nk * nm];

        let mut g_to_nm = vec![(0usize, 0usize); components];
        for n in 0..nb {
            for m in 0..nb {
                for g in svd.pair_range(n, m) {
                    g_to_nm[g] = (n, m);
                }
            }
        }

        wtab.par_chunks_mut(components)
            .zip(vtab.par_chunks_mut(nm * components))
            .enumerate()
            .for_each(|(index, (u_row, v_row))| {
                let point = Self::point_for(grid_n, index);
                let pe: Vec<Cf> = wigner.k.irvec.iter().map(|&r| phase(point, r)).collect();
                let pp: Vec<Cf> = wigner.g.irvec.iter().map(|&r| phase(point, r)).collect();
                for g in 0..components {
                    let (n, m) = g_to_nm[g];
                    for re in 0..epw.nrr_k {
                        let d = wigner.k.ndegen(re, n, m);
                        if d > 0 {
                            u_row[g] += pe[re] * svd.u_at(g, re) / d as f64;
                        }
                    }
                    for rp in 0..epw.nrr_g {
                        // let f = phase(point, wigner.g.irvec[rp]);
                        for nu in 0..nm {
                            for a in 0..nm {
                                let d = wigner.g.ndegen(rp, 0, a / 3);
                                if d > 0 {
                                    let e = phonon_evec[(index * nm + nu) * nm + a];
                                    v_row[nu * components + g] +=
                                        pp[rp] * svd.v_at(g, rp, a) * e / d as f64;
                                }
                            }
                        }
                    }
                }
            });

        let mut table = Self {
            grid_n,
            nbnd: nb,
            nmodes: nm,
            pair_offsets: svd.pair_offsets.clone(),
            s: svd.s.clone(),
            wtab,
            vtab,
            cbm_evec,
            electron_ev,
            phonon_ev,
            long_ev,
            phfreq_cutoff_ev,
            bra_band: vec![],
        };
        table.precontract();
        Ok(table)
    }

    pub fn len(&self) -> usize {
        self.grid_n.pow(3)
    }

    pub fn is_empty(&self) -> bool {
        self.grid_n == 0
    }

    pub fn point(&self, index: usize) -> Vector3<f64> {
        Self::point_for(self.grid_n, index)
    }

    fn point_for(n: usize, index: usize) -> Vector3<f64> {
        let i = index / (n * n);
        let j = index / n % n;
        let k = index % n;
        Vector3::new(
            i as f64 / n as f64,
            j as f64 / n as f64,
            k as f64 / n as f64,
        )
    }

    pub fn index_of(&self, point: Vector3<f64>) -> Option<usize> {
        let mut ijk = [0; 3];
        for axis in 0..3 {
            let scaled = point[axis] * self.grid_n as f64;
            let nearest = scaled.round();
            if (scaled - nearest).abs() > 1e-9 {
                return None;
            }
            ijk[axis] = (nearest as i64).rem_euclid(self.grid_n as i64) as usize;
        }
        Some((ijk[0] * self.grid_n + ijk[1]) * self.grid_n + ijk[2])
    }

    fn cbm(&self, k: usize) -> &[Cf] {
        &self.cbm_evec[k * self.nbnd..(k + 1) * self.nbnd]
    }

    pub fn short_indices(&self, k: usize, q: usize, nu: usize) -> Cf {
        self.short_indices_capped(k, q, nu, &usize::MAX)
    }

    /// `(k + q) mod grid`, purely on flat indices — the hot path must not
    /// round-trip through fractional coordinates.
    #[inline]
    pub fn wrap_add(&self, k: usize, q: usize) -> usize {
        let n = self.grid_n;
        let nn = n * n;
        let i = (k / nn + q / nn) % n;
        let j = (k / n % n + q / n % n) % n;
        let l = (k % n + q % n) % n;
        (i * n + j) * n + l
    }

    pub fn short_indices_capped(&self, k: usize, q: usize, nu: usize, rank_cap: &usize) -> Cf {
        debug_assert!(k < self.len() && q < self.len() && nu < self.nmodes);
        let kp = self.wrap_add(k, q);
        let ncomp = self.s.len();
        let u_row = &self.wtab[k * ncomp..(k + 1) * ncomp];
        let v_row = &self.vtab[(q * self.nmodes + nu) * ncomp..(q * self.nmodes + nu + 1) * ncomp];
        let bra = self.cbm(kp);
        let mut out = Cf::new(0.0, 0.0);
        if *rank_cap == usize::MAX {
            for g in 0..ncomp {
                out += bra[self.bra_band[g] as usize].conj() * u_row[g] * v_row[g];
            }
        } else {
            for pair in 0..self.nbnd * self.nbnd {
                let start = self.pair_offsets[pair];
                let end = start
                    .saturating_add(*rank_cap)
                    .min(self.pair_offsets[pair + 1]);
                for g in start..end {
                    out += bra[self.bra_band[g] as usize].conj() * u_row[g] * v_row[g];
                }
            }
        }
        out * RY_TO_EV
    }

    pub fn full_indices(&self, k: usize, q: usize, nu: usize) -> Cf {
        self.full_indices_capped(k, q, nu, &usize::MAX)
    }

    pub fn full_indices_capped(&self, k: usize, q: usize, nu: usize, rank_cap: &usize) -> Cf {
        let kp = self.wrap_add(k, q);
        let overlap: Cf = self
            .cbm(kp)
            .iter()
            .zip(self.cbm(k))
            .map(|(bra, ket)| bra.conj() * ket)
            .sum();
        self.short_indices_capped(k, q, nu, rank_cap) + overlap * self.long_ev[q * self.nmodes + nu]
    }

    pub fn full(&self, k: Vector3<f64>, q: Vector3<f64>, nu: usize) -> Cf {
        let ki = self
            .index_of(k)
            .expect("electron momentum is not on the vertex grid");
        let qi = self
            .index_of(q)
            .expect("phonon momentum is not on the vertex grid");
        self.full_indices(ki, qi, nu)
    }

    /// Validate compressed values against the direct oracle on random grid triples.
    pub fn max_direct_error(&self, direct: &DirectVertex<'_>, samples: usize, seed: u64) -> f64 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut worst: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for _ in 0..samples {
            let k = rng.gen_range(0..self.len());
            let q = rng.gen_range(1..self.len());
            let nu = rng.gen_range(0..self.nmodes);
            let expected = direct.full(self.point(k), self.point(q))[nu];
            scale = scale.max(expected.norm());
            worst = worst.max((self.full_indices(k, q, nu) - expected).norm());
        }
        worst / scale
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, serde_json::to_vec(self)?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut table: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        if table.bra_band.is_empty() && !table.s.is_empty() {
            table.precontract();
        }
        Ok(table)
    }

    fn precontract(&mut self) {
        let ncomp = self.s.len();
        self.bra_band = vec![0; ncomp];
        for n in 0..self.nbnd {
            for m in 0..self.nbnd {
                let pair = n * self.nbnd + m;
                for g in self.pair_offsets[pair]..self.pair_offsets[pair + 1] {
                    self.bra_band[g] = n as u8;
                    for k in 0..self.len() {
                        let ket = self.cbm_evec[k * self.nbnd + m];
                        self.wtab[k * ncomp + g] *= self.s[g] * ket;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epw::{EpmatWp, WsGrid};

    #[test]
    fn wrap_add_matches_the_fractional_round_trip() {
        let table = VertexTable {
            grid_n: 5,
            nbnd: 1,
            nmodes: 1,
            pair_offsets: vec![0, 0],
            s: vec![],
            wtab: vec![],
            bra_band: vec![],
            vtab: vec![],
            cbm_evec: vec![Cf::new(1.0, 0.0); 125],
            electron_ev: vec![0.0; 125],
            phonon_ev: vec![1.0; 125],
            long_ev: vec![Cf::new(0.0, 0.0); 125],
            phfreq_cutoff_ev: DEFAULT_PHFREQ_CUTOFF_EV,
        };
        for k in 0..table.len() {
            for q in 0..table.len() {
                let expected = table.index_of(table.point(k) + table.point(q)).unwrap();
                assert_eq!(table.wrap_add(k, q), expected);
            }
        }
    }

    #[test]
    fn exact_periodic_grid_lookup() {
        let table = VertexTable {
            grid_n: 4,
            nbnd: 1,
            nmodes: 1,
            pair_offsets: vec![0, 0],
            s: vec![],
            wtab: vec![],
            bra_band: vec![],
            vtab: vec![],
            cbm_evec: vec![Cf::new(1.0, 0.0); 64],
            electron_ev: vec![0.0; 64],
            phonon_ev: vec![1.0; 64],
            long_ev: vec![Cf::new(0.0, 0.0); 64],
            phfreq_cutoff_ev: DEFAULT_PHFREQ_CUTOFF_EV,
        };
        assert_eq!(table.index_of(Vector3::new(0.25, -0.25, 1.0)), Some(28));
        assert_eq!(table.point(28), Vector3::new(0.25, 0.75, 0.0));
        assert_eq!(table.index_of(Vector3::new(0.2, 0.0, 0.0)), None);
    }

    #[test]
    fn compressed_table_matches_direct_oracle() {
        let epw = EpwData {
            ef: 0.0,
            nbnd: 2,
            nrr_k: 1,
            nmodes: 3,
            nrr_q: 1,
            nrr_g: 1,
            zstar: vec![[[0.0; 3]; 3]],
            epsi: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            h: vec![
                Cf::new(0.0, 0.0),
                Cf::new(0.0, 0.0),
                Cf::new(0.0, 0.0),
                Cf::new(2.0, 0.0),
            ],
            d: vec![
                Cf::new(1.0, 0.0),
                Cf::new(0.0, 0.0),
                Cf::new(0.0, 0.0),
                Cf::new(0.0, 0.0),
                Cf::new(4.0, 0.0),
                Cf::new(0.0, 0.0),
                Cf::new(0.0, 0.0),
                Cf::new(0.0, 0.0),
                Cf::new(9.0, 0.0),
            ],
        };
        let crystal = Crystal {
            nat: 1,
            nmodes: 3,
            at: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            bg: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            omega: 100.0,
            alat: 5.0,
            tau: vec![[0.0; 3]],
            ityp: vec![1],
            amass: vec![1.0],
        };
        let wigner = Wigner {
            k: WsGrid::from_parts(vec![[0; 3]], 2, 2, vec![1; 4]),
            q: WsGrid::from_parts(vec![[0; 3]], 1, 1, vec![1]),
            g: WsGrid::from_parts(vec![[0; 3]], 2, 1, vec![1; 2]),
        };
        let epmat = EpmatWp {
            nbnd: 2,
            nrr_k: 1,
            nmodes: 3,
            nrr_g: 1,
            data: (0..12)
                .map(|i| Cf::new((i + 1) as f64, (i % 3) as f64 - 1.0))
                .collect(),
        };
        let svd = EphSvd::compress(&epmat, 1e-10, 3);
        assert!(svd.max_relative_error(&epmat) < 1e-9);
        let table =
            VertexTable::build(&epw, &crystal, &wigner, &svd, 2, DEFAULT_PHFREQ_CUTOFF_EV).unwrap();
        let direct = DirectVertex::new(&epw, &crystal, &wigner, &epmat);
        assert!(table.max_direct_error(&direct, 200, 19) < 1e-9);
    }

    #[test]
    fn precontracted_evaluation_matches_band_pair_loop() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(23);
        let grid_n = 2_usize;
        let nk = grid_n.pow(3);
        let nbnd = 2;
        let ncomp = 8;
        let mut table = VertexTable {
            grid_n,
            nbnd,
            nmodes: 1,
            pair_offsets: vec![0, 2, 4, 6, 8],
            s: (0..ncomp).map(|_| rng.gen_range(0.2..1.2)).collect(),
            wtab: (0..nk * ncomp)
                .map(|_| Cf::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
                .collect(),
            bra_band: vec![],
            vtab: (0..nk * ncomp)
                .map(|_| Cf::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
                .collect(),
            cbm_evec: (0..nk * nbnd)
                .map(|_| Cf::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
                .collect(),
            electron_ev: vec![0.0; nk],
            phonon_ev: vec![1.0; nk],
            long_ev: vec![Cf::new(0.0, 0.0); nk],
            phfreq_cutoff_ev: DEFAULT_PHFREQ_CUTOFF_EV,
        };
        let raw = table.wtab.clone();
        let mut cases = Vec::with_capacity(100);
        for _ in 0..100 {
            let k = rng.gen_range(0..nk);
            let q = rng.gen_range(0..nk);
            let cap = rng.gen_range(1..=3);
            let kp = table.index_of(table.point(k) + table.point(q)).unwrap();
            let mut expected = Cf::new(0.0, 0.0);
            for n in 0..nbnd {
                for m in 0..nbnd {
                    let pair = n * nbnd + m;
                    let end = (table.pair_offsets[pair] + cap).min(table.pair_offsets[pair + 1]);
                    for g in table.pair_offsets[pair]..end {
                        expected += table.s[g]
                            * table.cbm(kp)[n].conj()
                            * table.cbm(k)[m]
                            * raw[k * ncomp + g]
                            * table.vtab[q * ncomp + g];
                    }
                }
            }
            cases.push((k, q, cap, expected * RY_TO_EV));
        }

        table.precontract();
        for (k, q, cap, expected) in cases {
            assert!((table.short_indices_capped(k, q, 0, &cap) - expected).norm() < 1.0e-12);
        }
    }
}
