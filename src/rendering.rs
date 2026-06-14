use crate::support::{Point, Scale};
use dashu::float::{DBig, FBig};
use hsl::HSL;
use itertools::Itertools;
use num::Complex;
use std::fmt::Display;
use std::num::NonZeroUsize;
use std::str::FromStr;

#[derive(Copy, Clone)]
pub enum Units {}
#[derive(Copy, Clone)]
pub enum Pixels {}
pub type Origin = Point<FBig, Units>;
pub type View = Scale<f64, Pixels, Units>;

#[derive(Clone)]
pub struct CoordinatesBox {
    pub origin: Origin,
    pub view: View,
}

impl Display for CoordinatesBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{},{}|{:e}",
            self.origin.x.to_decimal().value(),
            self.origin.y.to_decimal().value(),
            self.view.inner
        )
    }
}

impl FromStr for CoordinatesBox {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (origin_x, rest) = s.split_at(s.chars().position(|c| c == ',').ok_or(())?);
        let rest = &rest[1..]; // remove ,
        let (origin_y, view) = rest.split_at(rest.chars().position(|c| c == '|').ok_or(())?);
        let view = &view[1..]; // remove |

        let origin_x = origin_x
            .parse::<DBig>()
            .map_err(|_| ())?
            .to_binary()
            .value();
        let origin_y = origin_y
            .parse::<DBig>()
            .map_err(|_| ())?
            .to_binary()
            .value();

        Ok(Self {
            origin: Point::new(origin_x, origin_y),
            view: Scale::new(view.parse().map_err(|_| ())?),
        })
    }
}

fn is_bad_value(v: &Complex<FBig>) -> bool {
    // implicitly also checks that neither is NaN
    let k: FBig = FBig::try_from(100_000.0).unwrap();
    !(-&k <= v.re && v.re <= k && -&k <= v.im && v.im <= k)
}

/// Working float type for perturbation (reference projection + delta
/// iteration). Set to f32 for GPU-precision testing; flip back to f64 to
/// restore full CPU precision.
pub type Pf = f64;

pub struct Orbit<F> {
    pub is_full: bool,
    pub orbit: Box<[Complex<F>]>,
}

pub fn calculate_orbit(x_0: Complex<FBig>, iterations: usize) -> (Orbit<FBig>, Orbit<Pf>) {
    let mut out = Vec::with_capacity(iterations + 1);
    let mut x_n = x_0.clone();
    out.push(x_n.clone());
    let mut is_full = true;
    for _ in 0..iterations {
        x_n = &x_n * &x_n + &x_0;
        if is_bad_value(&x_n) {
            is_full = false;
            break;
        }
        out.push(x_n.clone());
    }
    let out = out.into_boxed_slice();
    let out_pf = out
        .iter()
        .map(|c| Complex {
            re: c.re.to_f64().value(),
            im: c.im.to_f64().value(),
        })
        .collect_vec()
        .into_boxed_slice();
    (
        Orbit {
            is_full,
            orbit: out,
        },
        Orbit {
            is_full,
            orbit: out_pf,
        },
    )
}

pub fn check_orbit(orbit: &Orbit<Pf>) -> Result<Option<NonZeroUsize>, ()> {
    for (i, x) in orbit.orbit.iter().enumerate() {
        if x.re * x.re + x.im * x.im > 4.0 {
            // this is always Some(...), i + 1 can't be 0
            return Ok(NonZeroUsize::new(i + 1));
        }
    }
    if orbit.is_full { Ok(None) } else { Err(()) }
}

pub fn check_divergence_delta(
    ref_orbit: &Orbit<Pf>,
    delta: Complex<Pf>,
) -> Result<Option<NonZeroUsize>, ()> {
    // ref_orbit is [x_0, x_1, ...]
    // delta is delta_0
    // delta_{n+1} = 2 x_n delta_n + delta_n^2 + delta_0

    let delta_0 = delta;
    let mut delta = delta;
    for (i, &c) in ref_orbit.orbit.iter().enumerate() {
        let x = c + delta;
        if x.re * x.re + x.im * x.im > 4.0 {
            // this is always Some(...), i + 1 can't be 0
            return Ok(NonZeroUsize::new(i + 1));
        }

        delta = c * 2.0 * delta + delta * delta + delta_0;
    }

    if ref_orbit.is_full {
        Ok(None)
    } else {
        Err(())
    }
}

fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

pub fn val_to_color(val: Option<NonZeroUsize>) -> u32 {
    if let Some(val) = val {
        // really good color scheme
        // let hsl = HSL {
        //     h: (val.get() as f64 / 400.0).sin() * 180.0,
        //     s: 0.7,
        //     l: ((val.get() as f64 / 40.0).sin() * 0.3 + 0.4),
        // };

        let hsl = HSL {
            h: (val.get() as f64 / 1200.0).sin() * 180.0,
            s: 0.7,
            l: ((val.get() as f64 / 40.0).sin() * 0.3 + 0.4),
        };

        let (r, g, b) = hsl.to_rgb();
        rgb_to_u32(r, g, b)
    } else {
        0
    }
}

