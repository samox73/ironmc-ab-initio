//! Reader for a coarse-grid `.epb` file — EPW's Bloch-basis e-ph dump, one file
//! per pool, written *before* the long-range subtraction. So its coupling is the
//! **full** vertex (short + long), the real-data target for [`crate::longrange`].
//!
//! One unformatted Fortran record (byte-verified against `datasets/LiF`):
//! ```text
//!   [i4 marker=L] nqc(i4) xqc(3,nqc) et(nbnd,nks) dynq(nm,nm,nqc)
//!                 epmatq(nbndep,nbndep,nks,nm,nqc) zstar(3,3,nat) epsi(3,3) [i4 L]
//! ```
//! all little-endian f64 / `(re,im)` c16, Fortran column-major (first index
//! fastest). `nks` (pool-local k count) is recovered from the record length.
//! We keep the ~100 MB of `epmatq` as raw bytes and decode only the CBM slice on
//! demand — no eager parse. (patt126 assumed a mixed multi-record layout; wrong.)

use num_complex::Complex64;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct Epb {
    pub nqc: usize,
    pub nks: usize,
    pub nbnd: usize,
    pub nbndep: usize,
    pub nmodes: usize,
    /// q-points, Cartesian in units of 2π/alat (as EPW stores them).
    pub xqc: Vec<[f64; 3]>,
    raw: Vec<u8>,
    et_off: usize,
    dynq_off: usize,
    epmatq_off: usize,
}

fn f64_at(raw: &[u8], byte: usize) -> f64 {
    f64::from_le_bytes(raw[byte..byte + 8].try_into().unwrap())
}

impl Epb {
    /// `nbnd` = full nscf band count in `et`; `nbndep` = bands kept in `epmatq`;
    /// `nmodes` = phonon modes. (For LiF: 20, 15, 6.)
    pub fn read(path: impl AsRef<Path>, nbnd: usize, nbndep: usize, nmodes: usize) -> Result<Self> {
        let raw = std::fs::read(path)?;
        let marker = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
        if marker + 8 != raw.len() {
            return Err(
                format!("epb: record marker {marker} != file body {}", raw.len() - 8).into(),
            );
        }
        let nqc = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;

        // Recover nks from the record length (everything else is known).
        let fixed = 4 + 24 * nqc + 16 * nmodes * nmodes * nqc + 8 * 27; // nqc,xqc,dynq,zstar,epsi
        let per_k = 8 * nbnd + 16 * nbndep * nbndep * nmodes * nqc; // et + epmatq
        if !(marker - fixed).is_multiple_of(per_k) {
            return Err(format!("epb: length {marker} inconsistent with dims (nbnd={nbnd}, nbndep={nbndep}, nmodes={nmodes}, nqc={nqc})").into());
        }
        let nks = (marker - fixed) / per_k;

        let xqc_off = 8;
        let et_off = xqc_off + 24 * nqc;
        let dynq_off = et_off + 8 * nbnd * nks;
        let epmatq_off = dynq_off + 16 * nmodes * nmodes * nqc;

        let xqc = (0..nqc)
            .map(|iq| {
                let b = xqc_off + 24 * iq; // (3,nqc) col-major: component fastest
                [f64_at(&raw, b), f64_at(&raw, b + 8), f64_at(&raw, b + 16)]
            })
            .collect();

        Ok(Epb {
            nqc,
            nks,
            nbnd,
            nbndep,
            nmodes,
            xqc,
            raw,
            et_off,
            dynq_off,
            epmatq_off,
        })
    }

    /// Coarse dynamical matrix `dynq[a,b]` at q-point `iq`, flat row-major
    /// `nmodes×nmodes`. This is the **full** Cartesian force-constant matrix (not
    /// mass-divided, LO–TO splitting included) — divide by `√(M_a M_b)` before
    /// diagonalizing for frequencies.
    pub fn dynq_at(&self, iq: usize) -> Vec<Complex64> {
        let nm = self.nmodes;
        let mut out = vec![Complex64::new(0.0, 0.0); nm * nm];
        for b in 0..nm {
            for a in 0..nm {
                // dynq(a,b,iq) col-major, first index fastest.
                let idx = a + nm * (b + nm * iq);
                let byte = self.dynq_off + 16 * idx;
                out[a * nm + b] =
                    Complex64::new(f64_at(&self.raw, byte), f64_at(&self.raw, byte + 8));
            }
        }
        out
    }

    /// nscf eigenvalue (Ry) of band `ib` at pool-local k-point `ik`.
    pub fn et(&self, ik: usize, ib: usize) -> f64 {
        f64_at(&self.raw, self.et_off + 8 * (ib + self.nbnd * ik)) // (nbnd,nks) col-major
    }

    /// Pool-local k with the minimum energy in band `cbm_band` — this is Γ when
    /// the pool contains it (the CBM of LiF sits at Γ). `cbm_band` indexes `et`
    /// (LiF: 5, the 6th nscf band).
    pub fn argmin_k(&self, cbm_band: usize) -> usize {
        (0..self.nks)
            .min_by(|&a, &b| {
                self.et(a, cbm_band)
                    .partial_cmp(&self.et(b, cbm_band))
                    .unwrap()
            })
            .unwrap()
    }

    /// Full coupling `g(ik→ik+q)` for the CBM (kept-band local index 0), per
    /// Cartesian atomic-displacement mode, in Ry — same basis/units as
    /// `g_cart + g_long_cart`.
    pub fn g_cbm_cart(&self, ik: usize, iq: usize) -> Vec<Complex64> {
        let ne = self.nbndep;
        (0..self.nmodes)
            .map(|imode| {
                // epmatq(0,0,ik,imode,iq), col-major, first index fastest.
                let idx = ne * (ne * (ik + self.nks * (imode + self.nmodes * iq)));
                let b = self.epmatq_off + 16 * idx;
                Complex64::new(f64_at(&self.raw, b), f64_at(&self.raw, b + 8))
            })
            .collect()
    }
}
