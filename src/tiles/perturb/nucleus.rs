//! Reference-point selection: find a nearby hyperbolic-component nucleus.
//!
//! Perturbation against a reference that escapes leaves every pixel that
//! outlives it "glitched" (see `gpu.rs`).  A nucleus `c*` of period `p` has a
//! superattracting cycle (`z_p(c*) = 0`), so its orbit never escapes and a
//! single reference serves every pixel for any iteration count.
//!
//! The search, in the usual `z_0 = 0, z_{n+1} = z_n² + c` convention (the
//! renderer's `X_0 = C` orbit is the same sequence shifted by one):
//!
//! 1. **Ball period detection.**  Iterate the view centre `c₀` together with
//!    `dz/dc`.  To first order the disc `|c - c₀| < r` maps to the disc
//!    centred at `z_n` with radius `r·|dz_n|`; the first `n` where that disc
//!    contains 0 is the period of a nucleus inside (or near) the view.
//! 2. **Newton.**  Solve `z_p(c) = 0` from `c₀`: `c ← c - z_p(c) / z_p'(c)`.
//! 3. **Verify.**  The result must land near the view; the caller also checks
//!    that its orbit really is full.
//!
//! If nothing is found at the view's own radius the disc is widened a few
//! times; a nucleus slightly off-screen is still a far better reference than
//! an on-screen point that escapes.

use dashu::base::EstimatedLog2;
use dashu::float::FBig;
use num::Complex;

type C = Complex<FBig>;

/// Newton step cap.  Convergence is quadratic once in the basin, so this is
/// only hit when the start point is outside it.
const MAX_NEWTON_STEPS: usize = 64;

/// Stop Newton once the step is below 2^-NEWTON_TOL_BITS pixels (and below
/// the component's size, see `newton_nucleus`).
const NEWTON_TOL_BITS: i64 = 40;

/// Bits beyond `2·log2|dz_p/dc|` that Newton works at (see `newton_nucleus`).
const NUCLEUS_PREC_MARGIN: usize = 64;

/// Bits by which the last Newton step must be below the component's size.
const NUCLEUS_TOL_MARGIN: i64 = 20;

/// Largest step multiplier for Newton's walking phase (see `newton_nucleus`).
const NEWTON_MAX_MULT: u32 = 1 << 12;

/// Ball radii tried, as multiples of the view's half-diagonal.
const RADIUS_MULTIPLIERS: [u32; 3] = [1, 4, 16];

/// `|z|²` beyond which an orbit is considered escaped (2^64).  Also keeps
/// FBig exponents from doubling without bound once an orbit has escaped.
const ESCAPE_NORM2_LOG2: isize = 64;

/// How far (in view half-diagonals) a nucleus may be from the view centre and
/// still be used.  Pixel deltas grow with this distance and f32 deltas lose
/// relative precision, so keep it moderate.
pub const MAX_REF_DIST: u32 = 16;

#[derive(Debug)]
pub struct Interrupted;

pub struct Nucleus {
    pub c:      C,
    pub period: usize,
}

fn norm2(z: &C) -> FBig {
    &z.re * &z.re + &z.im * &z.im
}

fn at_prec(x: &FBig, prec: usize) -> FBig {
    x.clone().with_precision(prec).value()
}

/// `z² + c` with two multiplications.
fn sq_add(z: &C, c: &C) -> C {
    let re = (&z.re + &z.im) * (&z.re - &z.im) + &c.re;
    let im = ((&z.re * &z.im) << 1) + &c.im;
    Complex { re, im }
}

/// `dz/dc` as an f64 mantissa times `2^e`: it only needs relative accuracy,
/// and FBig arithmetic on it cost more than the orbit itself. (Its range is
/// unbounded; `z` enters as f64, which is fine while |z| > 2^-1022.)
struct Deriv {
    m: Complex<f64>,
    e: i64,
}

impl Deriv {
    fn zero() -> Self { Self { m: Complex { re: 0.0, im: 0.0 }, e: 0 } }

    /// `dz ← 2·z·dz + 1`.
    fn step(&mut self, z: Complex<f64>) {
        let one = (-self.e as f64).exp2();
        self.m = z * self.m * 2.0 + Complex { re: one, im: 0.0 };
        let a = self.m.re.abs().max(self.m.im.abs());
        if a != 0.0 && !(2f64.powi(-64)..2f64.powi(64)).contains(&a) {
            let k = a.log2().floor() as i64;
            self.m = self.m * (-k as f64).exp2();
            self.e += k;
        }
    }

    /// log2 |dz|².
    fn log2_norm2(&self) -> f64 {
        self.m.norm_sqr().log2() + 2.0 * self.e as f64
    }

    fn to_fbig(&self, prec: usize) -> C {
        let f = |x: f64| at_prec(&FBig::try_from(x).unwrap(), prec) << self.e as isize;
        Complex { re: f(self.m.re), im: f(self.m.im) }
    }
}

fn to_f64(z: &C) -> Complex<f64> {
    Complex { re: z.re.to_f64().value(), im: z.im.to_f64().value() }
}

/// Lowest `p ≤ max_period` such that the first-order image of the disc
/// `|c - center| < radius` under `z_p` contains 0, or `None`.
pub fn ball_period(
    center:     &C,
    radius:     &FBig,
    max_period: usize,
    prec:       usize,
    interrupted: &dyn Fn() -> bool,
) -> Result<Option<usize>, Interrupted> {
    let log2_r2 = 2.0 * radius.log2_est() as f64;
    let bail = (ESCAPE_NORM2_LOG2 as f64).exp2();
    let zero = at_prec(&FBig::ZERO, prec);
    let (mut z, mut dz) = (Complex { re: zero.clone(), im: zero }, Deriv::zero());
    for p in 1..=max_period {
        if p % 256 == 0 && interrupted() { return Err(Interrupted); }
        dz.step(to_f64(&z));
        z = sq_add(&z, center);
        let zn = to_f64(&z).norm_sqr();
        if zn > bail { return Ok(None); }
        if zn == 0.0 || zn.log2() < log2_r2 + dz.log2_norm2() { return Ok(Some(p)); }
    }
    Ok(None)
}

/// Newton-iterate towards a nucleus of period `period` starting at `guess`.
/// `tol2` is the squared step size at which to stop.  `None` if it diverges,
/// wanders further than `sqrt(max_dist2)` from `guess`, or fails to converge
/// within the step cap.
///
/// The component around a nucleus is only ~`1/|dz_p/dc|²` wide, which deep
/// in an embedded Julia set is far smaller than a pixel (at a 2^-325 view,
/// p≈49000: |dz_p/dc| ≈ 2^315, so ~2^-630). Its orbit stays bounded only
/// if `c` lies inside, so once Newton has converged at the view's precision
/// (`tol2`) it raises the precision to `2·log2|dz_p/dc| +
/// NUCLEUS_PREC_MARGIN` bits and converges to within the component (one or
/// two more, dearer, steps). At the view's precision alone it converged to
/// a point whose orbit escaped after little more than one period.
///
/// Far from the nucleus (relative to the structure around it) `z_p` looks
/// like `A·(c - c*)^k` for a large k, and a Newton step covers only 1/k of
/// the way: it walks with a nearly constant step (seen: 20+ steps at
/// ~200 ms each). So while consecutive steps point the same way the step is
/// multiplied by `mult`, doubling each time, and reset to 1 once a step
/// turns back (overshot); near the root the steps shrink and it is plain
/// Newton again.
pub fn newton_nucleus(
    guess:     &C,
    period:    usize,
    prec:      usize,
    tol2:      &FBig,
    max_dist2: &FBig,
    interrupted: &dyn Fn() -> bool,
) -> Result<Option<C>, Interrupted> {
    let bail = (ESCAPE_NORM2_LOG2 as f64).exp2();
    let mut prec = prec;
    let mut c = Complex { re: at_prec(&guess.re, prec), im: at_prec(&guess.im, prec) };
    let mut mult = 1u32;
    let mut last: Option<C> = None;
    for _ in 0..MAX_NEWTON_STEPS {
        let zero = at_prec(&FBig::ZERO, prec);
        let (mut z, mut dzf) = (Complex { re: zero.clone(), im: zero }, Deriv::zero());
        for i in 0..period {
            if i % 1024 == 1023 && interrupted() { return Err(Interrupted); }
            let zf = to_f64(&z);
            if zf.norm_sqr() > bail { return Ok(None); }
            dzf.step(zf);
            z = sq_add(&z, &c);
        }
        if dzf.m.norm_sqr() == 0.0 { return Ok(None); }
        let log2_d = dzf.log2_norm2().max(0.0) as f32;
        let dz = dzf.to_fbig(prec);
        let d = norm2(&dz);
        // z / dz = z · conj(dz) / |dz|²
        let num = Complex {
            re: &z.re * &dz.re + &z.im * &dz.im,
            im: &z.im * &dz.re - &z.re * &dz.im,
        };
        let delta = Complex { re: num.re / &d, im: num.im / &d };
        // Walking: this Newton step points the same way as the last one
        // (cos > 0.9) and is at least half its size.
        let walking = last.as_ref().is_some_and(|l| {
            let dot = &delta.re * &l.re + &delta.im * &l.im;
            let (dl, ll) = (norm2(&delta), norm2(l));
            dot.sign() == dashu::base::Sign::Positive
                && &dot * &dot > (&dl * &ll) * FBig::try_from(0.81).unwrap()
                && dl * FBig::from(4) > ll
        });
        mult = if walking { (mult * 2).min(NEWTON_MAX_MULT) } else { 1 };
        let m = FBig::from(mult);
        c = Complex { re: &c.re - &delta.re * &m, im: &c.im - &delta.im * &m };
        last = Some(delta.clone());
        if interrupted() { return Err(Interrupted); }
        let off = Complex { re: &c.re - &guess.re, im: &c.im - &guess.im };
        if norm2(&off) > *max_dist2 { return Ok(None); }
        let step2 = norm2(&delta);
        if step2 < *tol2 && mult == 1 {
            let size2: FBig = FBig::ONE >> (2 * (log2_d as i64 + NUCLEUS_TOL_MARGIN)) as isize;
            if step2 < size2 { return Ok(Some(c)); }
            let need = log2_d as usize + NUCLEUS_PREC_MARGIN;
            if need > prec {
                prec = need;
                c = Complex { re: at_prec(&c.re, prec), im: at_prec(&c.im, prec) };
            }
        }
    }
    Ok(None)
}

/// Find a nucleus near `center` usable as a perturbation reference for a view
/// of half-diagonal `radius` whose pixels are `2^upp_log2` units wide.
pub fn find_nucleus(
    center:     &C,
    radius:     &FBig,
    upp_log2:   i64,
    max_period: usize,
    prec:       usize,
    interrupted: &dyn Fn() -> bool,
) -> Result<Option<Nucleus>, Interrupted> {
    let tol     = FBig::ONE << (upp_log2 - NEWTON_TOL_BITS) as isize;
    let tol2    = &tol * &tol;
    let max_d   = radius * FBig::from(MAX_REF_DIST);
    let max_d2  = &max_d * &max_d;
    let mut last_period = None;
    for mult in RADIUS_MULTIPLIERS {
        let r = radius * FBig::from(mult);
        let Some(period) = ball_period(center, &r, max_period, prec, interrupted)? else { continue };
        // A wider ball often finds the same period again; Newton would too.
        if last_period == Some(period) { continue; }
        last_period = Some(period);
        let Some(c) = newton_nucleus(center, period, prec, &tol2, &max_d2, interrupted)? else { continue };
        return Ok(Some(Nucleus { c, period }));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rendering::calculate_orbit;

    const PREC: usize = 128;

    fn f(x: f64) -> FBig { FBig::try_from(x).unwrap().with_precision(PREC).value() }
    fn c(re: f64, im: f64) -> C { Complex { re: f(re), im: f(im) } }
    fn to_f64(z: &C) -> (f64, f64) { (z.re.to_f64().value(), z.im.to_f64().value()) }

    fn find(center: C, radius: f64, upp_log2: i64) -> Nucleus {
        find_nucleus(&center, &f(radius), upp_log2, 10_000, PREC, &|| false)
            .unwrap()
            .expect("nucleus")
    }

    #[test]
    fn period_2_nucleus() {
        let n = find(c(-1.01, 0.01), 0.05, -30);
        assert_eq!(n.period, 2);
        let (re, im) = to_f64(&n.c);
        assert!((re + 1.0).abs() < 1e-12 && im.abs() < 1e-12, "{re} {im}");
    }

    #[test]
    fn airplane_and_rabbit() {
        let n = find(c(-1.75, 0.0), 0.005, -30);
        assert_eq!(n.period, 3);
        assert!((to_f64(&n.c).0 + 1.754_877_666_246_692_7).abs() < 1e-12);

        let n = find(c(-0.12, 0.74), 0.01, -30);
        assert_eq!(n.period, 3);
        let (re, im) = to_f64(&n.c);
        assert!((re + 0.122_561_166_876_653_6).abs() < 1e-12, "{re}");
        assert!((im - 0.744_861_766_619_744_2).abs() < 1e-12, "{im}");
    }

    /// A deep view with no in-set pixel: the nucleus found must give a full
    /// (never-escaping) reference orbit.
    #[test]
    fn deep_view_gives_full_orbit() {
        let center = c(-0.743_643_887_037_151, 0.131_825_904_205_330);
        let t = std::time::Instant::now();
        let n = find(center, 1e-12, -50);
        eprintln!("period {} found in {:?}", n.period, t.elapsed());
        let (orbit, _) = calculate_orbit(n.c, 20_000);
        assert!(orbit.is_full);
    }

    /// Deep (radius 1e-40, 260 bits) view just beside the period-998
    /// nucleus found above: the search must find it again at full precision.
    #[test]
    fn very_deep_view_near_minibrot() {
        let p = 260;
        let fp = |x: f64| FBig::try_from(x).unwrap().with_precision(p).value();
        let shallow = find(c(-0.743_643_887_037_151, 0.131_825_904_205_330), 1e-12, -50);
        // Refine the nucleus to 260 bits so a 1e-40 view around it is exact.
        let guess = Complex { re: fp(0.0) + &shallow.c.re, im: fp(0.0) + &shallow.c.im };
        let exact = newton_nucleus(&guess, shallow.period, p, &(FBig::ONE << -480isize), &fp(1.0), &|| false)
            .unwrap()
            .unwrap();
        let center = Complex {
            re: &exact.re + &fp(3e-41),
            im: &exact.im - &fp(2e-41),
        };
        let t = std::time::Instant::now();
        let n = find_nucleus(&center, &fp(1e-40), -140, 20_000, p, &|| false)
            .unwrap()
            .expect("nucleus");
        eprintln!("period {} found in {:?}", n.period, t.elapsed());
        assert_eq!(n.period, shallow.period);
        let (orbit, _) = calculate_orbit(n.c, 20_000);
        assert!(orbit.is_full);
    }
}
