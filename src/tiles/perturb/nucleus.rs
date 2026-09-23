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

use dashu::float::FBig;
use num::{Complex, Zero};

type C = Complex<FBig>;

/// Newton step cap.  Convergence is quadratic once in the basin, so this is
/// only hit when the start point is outside it.
const MAX_NEWTON_STEPS: usize = 64;

/// Stop Newton once the step is below 2^-NEWTON_TOL_BITS pixels.
const NEWTON_TOL_BITS: i64 = 40;

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

fn escape_norm2() -> FBig {
    FBig::ONE << ESCAPE_NORM2_LOG2
}

/// One step of `(z, dz) ← (z² + c, 2·z·dz + 1)`.
fn step(z: &C, dz: &C, c: &C) -> (C, C) {
    let zdz = z * dz;
    let dz = Complex { re: (zdz.re << 1) + FBig::ONE, im: zdz.im << 1 };
    let z = z * z + c;
    (z, dz)
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
    let r2   = radius * radius;
    let bail = escape_norm2();
    let zero = at_prec(&FBig::ZERO, prec);
    let (mut z, mut dz) = (Complex { re: zero.clone(), im: zero.clone() }, Complex { re: zero.clone(), im: zero });
    for p in 1..=max_period {
        if p % 256 == 0 && interrupted() { return Err(Interrupted); }
        (z, dz) = step(&z, &dz, center);
        let zn = norm2(&z);
        if zn > bail { return Ok(None); }
        if zn < &r2 * norm2(&dz) { return Ok(Some(p)); }
    }
    Ok(None)
}

/// Newton-iterate towards a nucleus of period `period` starting at `guess`.
/// `tol2` is the squared step size at which to stop.  `None` if it diverges,
/// wanders further than `sqrt(max_dist2)` from `guess`, or fails to converge
/// within the step cap.
pub fn newton_nucleus(
    guess:     &C,
    period:    usize,
    prec:      usize,
    tol2:      &FBig,
    max_dist2: &FBig,
    interrupted: &dyn Fn() -> bool,
) -> Result<Option<C>, Interrupted> {
    let bail = escape_norm2();
    let mut c = Complex { re: at_prec(&guess.re, prec), im: at_prec(&guess.im, prec) };
    for _ in 0..MAX_NEWTON_STEPS {
        let zero = at_prec(&FBig::ZERO, prec);
        let (mut z, mut dz) = (Complex { re: zero.clone(), im: zero.clone() }, Complex { re: zero.clone(), im: zero });
        for i in 0..period {
            if i % 1024 == 1023 && interrupted() { return Err(Interrupted); }
            (z, dz) = step(&z, &dz, &c);
            if norm2(&z) > bail { return Ok(None); }
        }
        let d = norm2(&dz);
        if d.is_zero() { return Ok(None); }
        // z / dz = z · conj(dz) / |dz|²
        let num = Complex {
            re: &z.re * &dz.re + &z.im * &dz.im,
            im: &z.im * &dz.re - &z.re * &dz.im,
        };
        let delta = Complex { re: num.re / &d, im: num.im / &d };
        c = Complex { re: &c.re - &delta.re, im: &c.im - &delta.im };
        if interrupted() { return Err(Interrupted); }
        let off = Complex { re: &c.re - &guess.re, im: &c.im - &guess.im };
        if norm2(&off) > *max_dist2 { return Ok(None); }
        if norm2(&delta) < *tol2 { return Ok(Some(c)); }
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
