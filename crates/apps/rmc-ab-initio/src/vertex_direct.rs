//! Direct full-k Wannier-to-Bloch vertex used as the compression oracle.

use crate::epw::{Crystal, EpmatWp, EpwData, Wigner};
use crate::longrange::{g_long_cart, project_cartesian_branches};
use crate::reconstruct::{interpolate_d_polar, interpolate_h, phase};
use nalgebra::Vector3;
use num_complex::Complex64;

type Cf = Complex64;

pub struct DirectVertex<'a> {
    epw: &'a EpwData,
    crystal: &'a Crystal,
    wigner: &'a Wigner,
    epmat: &'a EpmatWp,
}

impl<'a> DirectVertex<'a> {
    pub fn new(
        epw: &'a EpwData,
        crystal: &'a Crystal,
        wigner: &'a Wigner,
        epmat: &'a EpmatWp,
    ) -> Self {
        assert_eq!(
            (epmat.nbnd, epmat.nrr_k, epmat.nmodes, epmat.nrr_g),
            (epw.nbnd, epw.nrr_k, epw.nmodes, epw.nrr_g)
        );
        Self {
            epw,
            crystal,
            wigner,
            epmat,
        }
    }

    /// Short-range Cartesian coupling in Ry, before the phonon-mode rotation.
    pub fn short_cart(&self, k: Vector3<f64>, q: Vector3<f64>) -> Vec<Cf> {
        let nb = self.epw.nbnd;
        let nm = self.epw.nmodes;
        let uk = &interpolate_h(self.epw, self.wigner, k).1[0];
        let kq = k + q;
        let ukq = &interpolate_h(self.epw, self.wigner, kq).1[0];
        let pe: Vec<Cf> = self.wigner.k.irvec.iter().map(|&r| phase(k, r)).collect();
        let pp: Vec<Cf> = self.wigner.g.irvec.iter().map(|&r| phase(q, r)).collect();
        let mut out = vec![Cf::new(0.0, 0.0); nm];

        for re in 0..self.epw.nrr_k {
            for n in 0..nb {
                for m in 0..nb {
                    let de = self.wigner.k.ndegen(re, n, m);
                    if de == 0 {
                        continue;
                    }
                    let electron = pe[re] * ukq[n].conj() * uk[m] / de as f64;
                    for rp in 0..self.epw.nrr_g {
                        for (a, value) in out.iter_mut().enumerate() {
                            let dp = self.wigner.g.ndegen(rp, 0, a / 3);
                            if dp > 0 {
                                *value +=
                                    electron * pp[rp] * self.epmat.g(re, rp, a, n, m) / dp as f64;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn project(&self, cart: &[Cf], phonons: &crate::reconstruct::Phonons) -> Vec<Cf> {
        project_cartesian_branches(
            self.crystal,
            phonons,
            cart,
            crate::vertex_table::DEFAULT_PHFREQ_CUTOFF_EV,
        )
    }

    /// Short-range coupling per physical phonon branch, in eV.
    pub fn short(&self, k: Vector3<f64>, q: Vector3<f64>) -> Vec<Cf> {
        let cart = self.short_cart(k, q);
        let phonons = interpolate_d_polar(self.epw, self.crystal, self.wigner, q);
        self.project(&cart, &phonons)
    }

    /// Full short-range plus polar coupling per branch, in eV.
    pub fn full_cart(&self, k: Vector3<f64>, q: Vector3<f64>) -> Vec<Cf> {
        let mut cart = self.short_cart(k, q);
        let uk = &interpolate_h(self.epw, self.wigner, k).1[0];
        let kq = k + q;
        let ukq = &interpolate_h(self.epw, self.wigner, kq).1[0];
        let overlap: Cf = ukq.iter().zip(uk).map(|(bra, ket)| bra.conj() * ket).sum();
        for (value, long) in cart.iter_mut().zip(g_long_cart(self.crystal, self.epw, q)) {
            *value += overlap * long;
        }
        cart
    }

    /// Full short-range plus polar coupling per branch, in eV.
    pub fn full(&self, k: Vector3<f64>, q: Vector3<f64>) -> Vec<Cf> {
        let cart = self.full_cart(k, q);
        let phonons = interpolate_d_polar(self.epw, self.crystal, self.wigner, q);
        self.project(&cart, &phonons)
    }
}
