//! Bilinear approximation (BLA, Zhuoran 2021): skip many perturbation
//! iterations at once while they are linear.
//!
//! With `δ_{n+1} = 2·X_n·δ_n + δ_n² + δ₀`, while |δ_n| is tiny next to
//! |2·X_n| the δ² term is negligible and a step is linear:
//! `δ ← A·δ + B·δ₀` with `A = 2·X_n`, `B = 1`. Linear steps compose, so a
//! block of `l` steps is again one `(A, B)`: block x then block y gives
//!
//! ```text
//! A = A_y·A_x      B = A_y·B_x + B_y
//! R = min(R_x, (R_y − |B_x|·max|δ₀|) / |A_x|)
//! ```
//!
//! where `R` is the validity radius: the block is accurate (relative error
//! below 2^BLA_EPS_LOG2) while |δ| < R when it starts. A single step is valid
//! while |δ|² ≤ ε·|2·X·δ|, i.e. `R = ε·|2·X|`.
//!
//! The table is a binary tree over the reference orbit: level k holds the
//! blocks of 2^k steps starting at multiples of 2^k, stored level after
//! level, each level padded to `2^(L−k)` entries (`L = log2_size`), so level
//! k starts at `2^(L+1) − 2^(L+1−k)`. A block that would need a reference
//! value past the end of the orbit is invalid.
//!
//! Coefficients overflow even f64 (A is a product of thousands of factors)
//! and radii underflow it at depth, so they are built in a small floatexp
//! (`Fx`) and uploaded as the shaders' floatexp: f32 mantissa + i32 exponent,
//! radius as log2.

use bytemuck::{Pod, Zeroable};
use num::Complex;

/// log2 of the relative error allowed per block (f32 mantissa precision).
pub const BLA_EPS_LOG2: f64 = -24.0;

/// log2 of an invalid radius. Finite on purpose: the shaders are compiled
/// with fast math, which may assume there are no infinities.
pub const INVALID_LOG2_R: f32 = -1.0e30;

/// Exponent of floatexp zero; must match FE_ZERO_EXP in floatexp.wgsl.
const FE_ZERO_EXP: i32 = -2_000_000_000;

/// One table entry, laid out as `struct Bla` in perturb_common.wgsl. The
/// vec2 fields come first: WGSL aligns vec2<f32> to 8 bytes, so any other
/// order would pad differently from this repr(C) struct (32 bytes both).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BlaEntry {
    pub a:      [f32; 2],
    pub b:      [f32; 2],
    pub a_e:    i32,
    pub b_e:    i32,
    pub log2_r: f32,
    _pad:       u32,
}

/// Complex floatexp with an f64 mantissa: `(re, im)·2^e`, the larger
/// component in [0.5, 1) (or zero).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fx {
    re: f64,
    im: f64,
    e:  i64,
}

impl Fx {
    const ZERO: Fx = Fx { re: 0.0, im: 0.0, e: i64::MIN / 4 };

    fn norm(re: f64, im: f64, e: i64) -> Fx {
        let mx = re.abs().max(im.abs());
        if mx == 0.0 || !mx.is_finite() {
            return Fx::ZERO;
        }
        let k = mx.log2().floor() as i64 + 1; // mx·2^-k in [0.5, 1)
        let s = (-k as f64).exp2();
        Fx { re: re * s, im: im * s, e: e + k }
    }

    pub fn from_c(z: Complex<f64>) -> Fx {
        Fx::norm(z.re, z.im, 0)
    }

    fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }

    fn mul(self, o: Fx) -> Fx {
        if self.is_zero() || o.is_zero() { return Fx::ZERO; }
        Fx::norm(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re, self.e + o.e)
    }

    fn add(self, o: Fx) -> Fx {
        if self.is_zero() { return o; }
        if o.is_zero() { return self; }
        let (hi, lo) = if self.e >= o.e { (self, o) } else { (o, self) };
        let s = ((lo.e - hi.e).max(-1100) as f64).exp2();
        Fx::norm(hi.re + lo.re * s, hi.im + lo.im * s, hi.e)
    }

    /// log2 |self|, −∞ for zero.
    pub fn log2_abs(self) -> f64 {
        if self.is_zero() { return f64::NEG_INFINITY; }
        0.5 * (self.re * self.re + self.im * self.im).log2() + self.e as f64
    }

    /// As the shaders' floatexp: f32 mantissa, i32 exponent.
    fn to_gpu(self) -> ([f32; 2], i32) {
        if self.is_zero() || self.e < FE_ZERO_EXP as i64 / 2 {
            return ([0.0, 0.0], FE_ZERO_EXP);
        }
        ([self.re as f32, self.im as f32], self.e.min(i32::MAX as i64 / 2) as i32)
    }

    #[cfg(test)]
    pub fn to_c(self) -> Complex<f64> {
        let s = (self.e as f64).exp2();
        Complex { re: self.re * s, im: self.im * s }
    }
}

/// One block: δ_after = A·δ + B·δ₀, valid while |δ| < 2^log2_r.
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub a:      Fx,
    pub b:      Fx,
    pub log2_r: f64,
}

impl Block {
    fn invalid() -> Block {
        Block { a: Fx::ZERO, b: Fx::ZERO, log2_r: f64::NEG_INFINITY }
    }

    /// The single step at reference value `x`.
    fn step(x: Complex<f64>) -> Block {
        let a = Fx::from_c(x * 2.0);
        Block { a, b: Fx::from_c(Complex { re: 1.0, im: 0.0 }), log2_r: BLA_EPS_LOG2 + a.log2_abs() }
    }

    /// `self` followed by `y`, for |δ₀| ≤ 2^log2_dc.
    fn then(self, y: Block, log2_dc: f64) -> Block {
        if self.log2_r == f64::NEG_INFINITY || y.log2_r == f64::NEG_INFINITY {
            return Block::invalid();
        }
        let a = y.a.mul(self.a);
        let b = y.a.mul(self.b).add(y.b);
        // R_y − |B_x|·|δ₀|, in log space.
        let t = self.b.log2_abs() + log2_dc;
        let log2_r = if t >= y.log2_r {
            f64::NEG_INFINITY
        } else {
            let room = y.log2_r + (1.0 - (t - y.log2_r).exp2()).log2();
            self.log2_r.min(room - self.a.log2_abs())
        };
        Block { a, b, log2_r }
    }

    fn to_gpu(self) -> BlaEntry {
        let (a, a_e) = self.a.to_gpu();
        let (b, b_e) = self.b.to_gpu();
        let log2_r = if self.log2_r.is_finite() { self.log2_r as f32 } else { INVALID_LOG2_R };
        BlaEntry { a, b, a_e, b_e, log2_r, _pad: 0 }
    }
}

/// The BLA table for one reference orbit and one bound on |δ₀|.
pub struct BlaTable {
    /// log2 of the level-0 size (padded orbit length).
    pub log2_size: u32,
    /// Number of levels (level k: blocks of 2^k steps).
    pub levels:    u32,
    pub entries:   Vec<BlaEntry>,
    /// The same blocks on the CPU, for tests.
    #[cfg(test)]
    pub blocks:    Vec<Block>,
}

impl BlaTable {
    /// Index of block `i` of level `k`.
    pub fn index(log2_size: u32, k: u32, i: usize) -> usize {
        (1usize << (log2_size + 1)) - (1usize << (log2_size + 1 - k)) + i
    }

    /// Build from the reference orbit `x` (X_0, X_1, …) for pixels with
    /// |δ₀| ≤ 2^log2_dc. Step m (from X_m to X_{m+1}) needs m + 1 < len.
    pub fn build(x: &[Complex<f64>], log2_dc: f64) -> BlaTable {
        let steps = x.len().saturating_sub(1).max(1);
        let log2_size = steps.next_power_of_two().trailing_zeros();
        let levels = log2_size + 1;
        let mut blocks = vec![Block::invalid(); Self::index(log2_size, levels - 1, 0) + 1];
        for (m, &xm) in x.iter().take(steps).enumerate() {
            blocks[Self::index(log2_size, 0, m)] = Block::step(xm);
        }
        for k in 1..levels {
            for i in 0..(1usize << (log2_size - k)) {
                let lo = blocks[Self::index(log2_size, k - 1, 2 * i)];
                let hi = blocks[Self::index(log2_size, k - 1, 2 * i + 1)];
                blocks[Self::index(log2_size, k, i)] = lo.then(hi, log2_dc);
            }
        }
        BlaTable {
            log2_size,
            levels,
            entries: blocks.iter().map(|b| b.to_gpu()).collect(),
            #[cfg(test)]
            blocks,
        }
    }

    /// A table that never applies (BLA disabled): one invalid entry.
    pub fn disabled() -> BlaTable {
        BlaTable {
            log2_size: 0,
            levels: 0,
            entries: vec![Block::invalid().to_gpu()],
            #[cfg(test)]
            blocks: vec![Block::invalid()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fx(re: f64, im: f64) -> Fx { Fx::from_c(Complex { re, im }) }

    #[test]
    fn fx_arithmetic() {
        let a = fx(3.0, -4.0);
        assert!((a.log2_abs() - 5f64.log2()).abs() < 1e-12);
        let p = a.mul(fx(0.5, 2.0)).to_c();
        let q = Complex { re: 3.0, im: -4.0 } * Complex { re: 0.5, im: 2.0 };
        assert!((p - q).norm() < 1e-12);
        let s = a.add(fx(1e-3, 1e-3)).to_c();
        assert!((s - Complex { re: 3.001, im: -3.999 }).norm() < 1e-12);
        // Far beyond f64 range and back.
        let mut big = fx(1.5, 0.0);
        for _ in 0..3000 { big = big.mul(fx(1.5, 0.0)); }
        assert!((big.log2_abs() - 3001.0 * 1.5f64.log2()).abs() < 1e-6);
    }

    /// Must match `struct Bla` in perturb_common.wgsl (wgpu rejects a
    /// size mismatch only at dispatch time, on a GPU).
    #[test]
    fn entry_layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<BlaEntry>(), 32);
        assert_eq!(std::mem::offset_of!(BlaEntry, b), 8);
        assert_eq!(std::mem::offset_of!(BlaEntry, a_e), 16);
        assert_eq!(std::mem::offset_of!(BlaEntry, log2_r), 24);
    }

    #[test]
    fn table_layout() {
        assert_eq!(BlaTable::index(3, 0, 0), 0);
        assert_eq!(BlaTable::index(3, 1, 0), 8);
        assert_eq!(BlaTable::index(3, 2, 0), 12);
        assert_eq!(BlaTable::index(3, 3, 0), 14);
        let t = BlaTable::build(&[Complex { re: 0.1, im: 0.2 }; 9], -40.0);
        assert_eq!((t.log2_size, t.levels, t.entries.len()), (3, 4, 15));
    }

    /// A block jump equals stepping, within the error the radius promises:
    /// real orbit (the period-3 "rabbit" nucleus, never escapes), δ at the
    /// block's radius / 4, δ₀ at its bound.
    #[test]
    fn jump_matches_stepping() {
        let c = Complex { re: -0.122_561_166_876_653_6, im: 0.744_861_766_619_744_2 };
        let mut x = vec![c];
        for _ in 0..4096 { let z = *x.last().unwrap(); x.push(z * z + c); }
        let log2_dc = -60.0;
        let t = BlaTable::build(&x, log2_dc);
        let mut checked = 0;
        for k in 1..t.levels {
            for i in 0..(1usize << (t.log2_size - k)).min(64) {
                let blk = t.blocks[BlaTable::index(t.log2_size, k, i)];
                if !blk.log2_r.is_finite() { continue; }
                let m = i << k;
                let d0 = Complex { re: log2_dc.exp2() * 0.6, im: log2_dc.exp2() * 0.8 };
                let r = (blk.log2_r - 2.0).exp2();
                let mut d = Complex { re: r * 0.8, im: -r * 0.6 };
                let start = d;
                for s in 0..(1usize << k) { d = x[m + s] * 2.0 * d + d * d + d0; }
                let jump = blk.a.mul(Fx::from_c(start)).add(blk.b.mul(Fx::from_c(d0))).to_c();
                let err = (jump - d).norm() / d.norm();
                assert!(err < 1e-5, "level {k} block {i}: relative error {err}");
                checked += 1;
            }
        }
        assert!(checked > 20, "only {checked} valid blocks");
    }
}
