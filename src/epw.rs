//! Readers for the three EPW artifacts a DiagMC run needs from a completed
//! `ephwann_shuffle`: `epwdata.fmt`, `crystal.fmt`, and the binary `epmatwp`.
//!
//! Layouts are taken from EPW's `epw_write`/`crystal_write` and cross-checked
//! against patt126's `IO.py` (reference only, not copied). All three are little-
//! endian; the `.fmt` files are Fortran list-directed text (`E` exponents,
//! complex printed as `(re,im)`), the `.epmatwp` is a raw `(f64 re, f64 im)`
//! stream with **no** Fortran record markers.

use num_complex::Complex64;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Header + Wannier-rep operators from `epwdata.fmt`.
///
/// `h`/`d` hold the raw complex streams; index them via [`EpwData::h_at`] /
/// [`EpwData::d_at`], which encode EPW's write loop order (real-space vector
/// varies fastest).
pub struct EpwData {
    /// Fermi energy (Ry).
    pub ef: f64,
    pub nbnd: usize,
    /// # Wigner-Seitz vectors on the electron (H) grid — the H(R_e) axis.
    pub nrr_k: usize,
    pub nmodes: usize,
    /// # WS vectors on the phonon (D) grid.
    pub nrr_q: usize,
    /// # WS vectors on the el-ph grid — the R_p axis of `epmatwp`.
    pub nrr_g: usize,
    /// Born effective charges Z*[atom][row][col] (nat = nmodes/3 atoms).
    pub zstar: Vec<[[f64; 3]; 3]>,
    /// High-frequency dielectric tensor ε∞[row][col].
    pub epsi: [[f64; 3]; 3],
    /// Wannier Hamiltonian H(R_e), stream order (band_i, band_j, R_e), R_e fastest.
    pub h: Vec<Complex64>,
    /// Wannier dynamical matrix D(R_p), stream order (mode_a, mode_b, R_q), R_q fastest.
    pub d: Vec<Complex64>,
}

impl EpwData {
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let (header, mut cx) = split_epwdata(&raw);

        // header: ef, then 5 int dims, then the polar block (3*nmodes Z* reals + 9 ε∞).
        if header.len() < 6 {
            return Err(format!("epwdata.fmt: short header ({} tokens)", header.len()).into());
        }
        let ef: f64 = header[0].parse()?;
        let nbnd: usize = header[1].parse()?;
        let nrr_k: usize = header[2].parse()?;
        let nmodes: usize = header[3].parse()?;
        let nrr_q: usize = header[4].parse()?;
        let nrr_g: usize = header[5].parse()?;
        let nat = nmodes / 3;

        let polar: Vec<f64> = header[6..6 + 9 * nat + 9]
            .iter()
            .map(|t| t.parse())
            .collect::<std::result::Result<_, _>>()?;
        // Fortran column-major: Z*(3,3,nat) then ε∞(3,3), first index fastest.
        let mut zstar = vec![[[0.0; 3]; 3]; nat];
        for (k, z) in zstar.iter_mut().enumerate() {
            for b in 0..3 {
                for a in 0..3 {
                    z[a][b] = polar[a + 3 * b + 9 * k];
                }
            }
        }
        let mut epsi = [[0.0; 3]; 3];
        for b in 0..3 {
            for a in 0..3 {
                epsi[a][b] = polar[9 * nat + a + 3 * b];
            }
        }

        // Complex stream: H(R_e) [nrr_k*nbnd*nbnd] then D(R_p) [nrr_q*nmodes*nmodes].
        let nh = nrr_k * nbnd * nbnd;
        let nd = nrr_q * nmodes * nmodes;
        if cx.len() != nh + nd {
            return Err(format!(
                "epwdata.fmt: expected {} complex entries (H {} + D {}), found {}",
                nh + nd,
                nh,
                nd,
                cx.len()
            )
            .into());
        }
        let d = cx.split_off(nh);
        let h = cx;

        Ok(EpwData {
            ef,
            nbnd,
            nrr_k,
            nmodes,
            nrr_q,
            nrr_g,
            zstar,
            epsi,
            h,
            d,
        })
    }

    /// H(R_e)[i,j] at electron WS vector `r`.
    pub fn h_at(&self, r: usize, i: usize, j: usize) -> Complex64 {
        self.h[(i * self.nbnd + j) * self.nrr_k + r]
    }

    /// D(R_q)[a,b] at phonon WS vector `r`.
    pub fn d_at(&self, r: usize, a: usize, b: usize) -> Complex64 {
        self.d[(a * self.nmodes + b) * self.nrr_q + r]
    }
}

/// Split `epwdata.fmt` into whitespace header tokens (everything outside `(...)`)
/// and the flat complex stream (every `(re,im)` group), in file order. Robust to
/// however Fortran wrapped its list-directed records across lines.
fn split_epwdata(raw: &str) -> (Vec<String>, Vec<Complex64>) {
    let mut header = Vec::new();
    let mut cx = Vec::new();
    let mut rest = raw;
    while let Some(open) = rest.find('(') {
        header.extend(rest[..open].split_whitespace().map(str::to_owned));
        rest = &rest[open + 1..];
        let close = rest.find(')').expect("epwdata.fmt: unterminated complex");
        let (re, im) = rest[..close]
            .split_once(',')
            .expect("epwdata.fmt: malformed complex");
        cx.push(Complex64::new(
            re.trim().parse().expect("epwdata.fmt: bad real part"),
            im.trim().parse().expect("epwdata.fmt: bad imag part"),
        ));
        rest = &rest[close + 1..];
    }
    header.extend(rest.split_whitespace().map(str::to_owned));
    (header, cx)
}

/// Cell geometry from `crystal.fmt`. One EPW `WRITE` per line; we read the first
/// ten (through `ityp`), which is everything Phase A needs.
pub struct Crystal {
    pub nat: usize,
    pub nmodes: usize,
    /// Lattice vectors as columns `at[j]` (cartesian, in units of `alat`).
    pub at: [[f64; 3]; 3],
    /// Reciprocal lattice, same convention.
    pub bg: [[f64; 3]; 3],
    /// Cell volume (bohr³).
    pub omega: f64,
    /// Lattice parameter (bohr).
    pub alat: f64,
    /// Atomic positions τ (cartesian, in units of `alat`).
    pub tau: Vec<[f64; 3]>,
    /// Atom type index (1-based, as written).
    pub ityp: Vec<usize>,
    /// Per-type masses (Ry a.u.); index with `ityp[iat] - 1`, or use [`Crystal::mass`].
    pub amass: Vec<f64>,
}

impl Crystal {
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let mut lines = raw.lines();
        let mut next = || -> Result<Vec<f64>> {
            let line = lines.next().ok_or("crystal.fmt: unexpected EOF")?;
            Ok(line
                .split_whitespace()
                .map(|t| t.parse())
                .collect::<std::result::Result<_, _>>()?)
        };

        let nat = next()?[0] as usize;
        let nmodes = next()?[0] as usize;
        let _nelec_nbndskip = next()?; // nelec, nbndskip — unused in Phase A
        let at = triple3(&next()?);
        let bg = triple3(&next()?);
        let omega = next()?[0];
        let alat = next()?[0];
        let tau_flat = next()?;
        let amass = next()?;
        let ityp: Vec<usize> = next()?.iter().map(|&x| x as usize).collect();

        let tau = (0..nat)
            .map(|k| [tau_flat[3 * k], tau_flat[3 * k + 1], tau_flat[3 * k + 2]])
            .collect();
        Ok(Crystal {
            nat,
            nmodes,
            at,
            bg,
            omega,
            alat,
            tau,
            ityp,
            amass,
        })
    }

    /// Mass of atom `iat` (Ry a.u.).
    pub fn mass(&self, iat: usize) -> f64 {
        self.amass[self.ityp[iat] - 1]
    }
}

/// 9 column-major reals → `m[col]` = the col-th 3-vector.
fn triple3(v: &[f64]) -> [[f64; 3]; 3] {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

/// The e-ph Wannier tensor from binary `epmatwp` (Fortran layout
/// `(nbnd, nbnd, nrr_k, nmodes, nrr_g)`, first index fastest, no record markers).
pub struct EpmatWp {
    pub nbnd: usize,
    pub nrr_k: usize,
    pub nmodes: usize,
    pub nrr_g: usize,
    /// Flat column-major stream; index via [`EpmatWp::g`].
    pub data: Vec<Complex64>,
}

impl EpmatWp {
    pub fn expected_len(nbnd: usize, nrr_k: usize, nmodes: usize, nrr_g: usize) -> usize {
        nbnd * nbnd * nrr_k * nmodes * nrr_g
    }

    pub fn read(
        path: impl AsRef<Path>,
        nbnd: usize,
        nrr_k: usize,
        nmodes: usize,
        nrr_g: usize,
    ) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let n = Self::expected_len(nbnd, nrr_k, nmodes, nrr_g);
        if bytes.len() != n * 16 {
            return Err(format!(
                "epmatwp: expected {} bytes for dims {nbnd}×{nbnd}×{nrr_k}×{nmodes}×{nrr_g}, found {}",
                n * 16,
                bytes.len()
            )
            .into());
        }
        let mut data = Vec::with_capacity(n);
        for chunk in bytes.as_chunks::<16>().0 {
            let re = f64::from_le_bytes(chunk[..8].try_into().unwrap());
            let im = f64::from_le_bytes(chunk[8..].try_into().unwrap());
            data.push(Complex64::new(re, im));
        }
        Ok(EpmatWp {
            nbnd,
            nrr_k,
            nmodes,
            nrr_g,
            data,
        })
    }

    /// g_wann(R_e, R_p, mode, n, m). `re ∈ [0,nrr_k)`, `rp ∈ [0,nrr_g)`.
    pub fn g(&self, re: usize, rp: usize, mode: usize, n: usize, m: usize) -> Complex64 {
        let nb = self.nbnd;
        self.data[n + nb * (m + nb * (re + self.nrr_k * (mode + self.nmodes * rp)))]
    }
}

/// One Wigner-Seitz grid: integer R-vectors plus their `d1×d2` degeneracy
/// weights. EPW resolves degeneracy per orbital/atom pair (and some are 0,
/// meaning "this R does not contribute to this pair" — callers must skip those).
pub struct WsGrid {
    /// R-vectors as integer coefficients of the primitive lattice vectors.
    pub irvec: Vec<[i64; 3]>,
    pub d1: usize,
    pub d2: usize,
    ndegen: Vec<i32>, // [nr * d1 * d2], row-major (ir, a, b)
}

impl WsGrid {
    #[cfg(test)]
    pub(crate) fn from_parts(irvec: Vec<[i64; 3]>, d1: usize, d2: usize, ndegen: Vec<i32>) -> Self {
        assert_eq!(ndegen.len(), irvec.len() * d1 * d2);
        Self {
            irvec,
            d1,
            d2,
            ndegen,
        }
    }

    pub fn nr(&self) -> usize {
        self.irvec.len()
    }
    /// Degeneracy for R-vector `ir`, pair `(a, b)`.
    pub fn ndegen(&self, ir: usize, a: usize, b: usize) -> i32 {
        self.ndegen[(ir * self.d1 + a) * self.d2 + b]
    }
}

/// The three WS grids from `wigner.fmt`: `k` (electron, dims×dims), `q` (phonon,
/// nat×nat), `g` (el-ph, dims×nat). The three vector lists differ — the plan's
/// key correctness note (`g`'s R_e axis uses `k`, its R_p axis uses `g`).
pub struct Wigner {
    pub k: WsGrid,
    pub q: WsGrid,
    pub g: WsGrid,
}

impl Wigner {
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let mut lines = raw.lines();
        let hdr: Vec<usize> = lines
            .next()
            .ok_or("wigner.fmt: empty")?
            .split_whitespace()
            .map(|t| t.parse())
            .collect::<std::result::Result<_, _>>()?;
        let (nrr_k, nrr_q, nrr_g, dims, dims2) = (hdr[0], hdr[1], hdr[2], hdr[3], hdr[4]);
        let k = read_ws_block(&mut lines, nrr_k, dims, dims)?;
        let q = read_ws_block(&mut lines, nrr_q, dims2, dims2)?;
        let g = read_ws_block(&mut lines, nrr_g, dims, dims2)?;
        Ok(Wigner { k, q, g })
    }
}

/// One WS block: `nr` entries, each an `irvec(3) + wslen` header line (wslen
/// discarded) followed by `d1` lines of `d2` degeneracy integers.
fn read_ws_block<'a>(
    lines: &mut impl Iterator<Item = &'a str>,
    nr: usize,
    d1: usize,
    d2: usize,
) -> Result<WsGrid> {
    let mut irvec = Vec::with_capacity(nr);
    let mut ndegen = Vec::with_capacity(nr * d1 * d2);
    for _ in 0..nr {
        let head: Vec<&str> = lines
            .next()
            .ok_or("wigner.fmt: truncated")?
            .split_whitespace()
            .collect();
        irvec.push([head[0].parse()?, head[1].parse()?, head[2].parse()?]);
        for _ in 0..d1 {
            let row = lines.next().ok_or("wigner.fmt: truncated ndegen")?;
            for tok in row.split_whitespace() {
                ndegen.push(tok.parse()?);
            }
        }
    }
    if ndegen.len() != nr * d1 * d2 {
        return Err(format!(
            "wigner.fmt: block expected {} degen ints, got {}",
            nr * d1 * d2,
            ndegen.len()
        )
        .into());
    }
    Ok(WsGrid {
        irvec,
        d1,
        d2,
        ndegen,
    })
}
