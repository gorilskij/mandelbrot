use crate::drawing::maybe_pixel::MaybePixel;
use crossbeam_channel;
use euclid::{Length, Point2D, Scale};
use hsl::HSL;
use indicatif::ProgressBar;
use itertools::iproduct;
use num::{Complex, Float, ToPrimitive};
use num_bigfloat::BigFloat;
use palette::rgb::Rgb;
use palette::{Mix, Srgb, rgb};
use parking_lot::RwLock;
use rayon::ThreadPool;
use std::cmp::min;
use std::num::NonZeroUsize;
use std::ops::Range;
use waker_interrupter::MultiInterrupter;

pub enum Units {}
pub enum Pixels {}
pub enum Relative {}
pub type Point<U> = Point2D<f64, U>;
pub type Origin = Point2D<BigFloat, Units>;
pub type View = Scale<f64, Pixels, Units>;

#[derive(Copy, Clone, PartialEq)]
pub struct CoordinatesBox {
    pub origin: Origin,
    pub view: View,
}

trait TypedXY<T, U> {
    fn x(&self) -> Length<T, U>;
    fn y(&self) -> Length<T, U>;
}

impl<T, U> TypedXY<T, U> for Point2D<T, U>
where
    T: Copy,
{
    fn x(&self) -> Length<T, U> {
        Length::new(self.x)
    }

    fn y(&self) -> Length<T, U> {
        Length::new(self.y)
    }
}

// fn calculate(c: Complex<f64>, iterations: usize) -> Option<NonZeroUsize> {
//     let mut z: Complex<f64> = Complex::ZERO;
//     for i in 1..iterations + 1 {
//         z = z * z + c;
//         if z.norm() > 4.0 {
//             // SAFETY: i starts iterating at 1, it is never 0
//             unsafe { return Some(NonZeroUsize::new_unchecked(i)) };
//         }
//     }
//     None
// }

fn calculate_orbit<F: Float>(x_0: Complex<F>, iterations: usize) -> Vec<Complex<F>> {
    let mut out = Vec::with_capacity(iterations + 1);
    let mut x_n = x_0;
    out.push(x_n);
    for _ in 0..iterations {
        x_n = x_n * x_n + x_0;
        out.push(x_n);
    }
    out
}

fn is_bad_value(v: Complex<f64>) -> bool {
    // implicitly also checks that neither is NaN
    !(v.re.abs() <= 1000.0 && v.im.abs() <= 1000.0)
}

fn check_orbit<F: Float>(orbit: &[Complex<F>]) -> Option<NonZeroUsize> {
    let four = F::from(4.0).unwrap();
    for (i, x) in orbit.iter().enumerate() {
        if x.norm() > four {
            // this is always Some(...), i + 1 can't be 0
            return NonZeroUsize::new(i + 1);
        }
    }
    None
}

fn check_divergence_delta(
    ref_orbit: &[Complex<BigFloat>],
    ref_orbit_f64: &[Complex<f64>],
    delta: Complex<f64>,
) -> Result<Option<NonZeroUsize>, ()> {
    // ref_orbit is [x_0, x_1, ...]
    // delta is delta_0
    // delta_{n+1} = 2 x_n delta_n + delta_n^2 + delta_0

    let delta_0 = delta;
    let mut delta = delta;
    for i in 0..ref_orbit.len() {
        if is_bad_value(ref_orbit_f64[i]) {
            return Err(());
        }

        let x = ref_orbit_f64[i] + delta;
        if x.norm() > 4.0 {
            // this is always Some(...), i + 1 can't be 0
            return Ok(NonZeroUsize::new(i + 1));
        }

        delta = ref_orbit_f64[i] * 2.0 * delta + delta * delta + delta_0;
    }
    Ok(None)
}

fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

fn render_pixel(val: Option<NonZeroUsize>) -> u32 {
    if let Some(val) = val {
        // [0, 1)
        // let f = 1.0 - 1.0 / (val.get() as f64 / 100.0 + 1.0);

        let hsl = HSL {
            // h: val.get() as f64 % 360.0,
            // s: 0.7,
            // // l: 0.5,
            // l: (val.get() as f64 / 10.0).sin() * 0.1 + 0.5,
            h: (val.get() as f64 / 10.0).sin() * 180.0,
            s: 0.7,
            // l: 0.5,
            l: ((val.get() as f64 / (10.0 * std::f64::consts::E)).sin() * 0.3 + 0.4),
        };

        let (r, g, b) = hsl.to_rgb();
        rgb_to_u32(r, g, b)
    } else {
        0
    }
}

fn interpolate(
    color_tl: u32,
    color_tr: u32,
    color_bl: u32,
    color_br: u32,
    /* (y, x) where tl is (0, 0) */
    location: (f64, f64),
) -> u32 {
    let color_tl = Rgb::<Srgb, _>::from(color_tl).into_format();
    let color_tr = Rgb::<Srgb, _>::from(color_tr).into_format();
    let color_bl = Rgb::<Srgb, _>::from(color_bl).into_format();
    let color_br = Rgb::<Srgb, _>::from(color_br).into_format();

    let top = color_tl.mix(color_tr, location.1);
    let bottom = color_bl.mix(color_br, location.1);

    let color = top.mix(bottom, location.0).into_format();

    color.into_u32::<rgb::channels::Argb>()
}

pub fn sample_zoomed(
    src: &[u32],
    dest: &mut [u32],
    // src and dest must match
    width: usize,
    height: usize,
    //
    src_coords: CoordinatesBox,
    dest_coords: CoordinatesBox,
) {
    let fwidth = width as f64;
    let fheight = height as f64;

    let CoordinatesBox {
        origin: src_origin,
        view: src_view,
    } = src_coords;
    let CoordinatesBox {
        origin: dest_origin,
        view: dest_view,
    } = dest_coords;

    // src_origin and dest_origin are both absolute
    // calculate dest_origin in the [0, 1] reference frame given by src
    // let src_view_bf = Scale::<_, Pixels, Units>::new(BigFloat::from(src_view.0));
    // TODO: check numerical properties at high zoom
    let dest_origin_rel_src = {
        let p = ((dest_origin - src_origin).to_f64() / src_view).to_point();
        Point::<Relative>::new(p.x / fwidth, p.y / fheight)
    };

    for dest_row in 0..height {
        for dest_col in 0..width {
            // [0, 1] coordinates relative to the reference frame given by dest
            let point_rel_dest =
                Point::<Relative>::new(dest_col as f64 / fwidth, dest_row as f64 / fheight);

            // calculate [0, 1] coordinates in the reference frame given by src
            let point_rel_src = (dest_origin_rel_src
                + point_rel_dest.to_vector() * (dest_view.0 / src_view.0))
                .clamp(
                    Point::zero(),
                    Point::new((fwidth - 1.0) / fwidth, (fheight - 1.0) / fheight),
                );

            let tl = (
                (point_rel_src.y * fheight) as usize,
                (point_rel_src.x * fwidth) as usize,
            );
            let tr = (
                (point_rel_src.y * fheight) as usize,
                (point_rel_src.x * fwidth).ceil() as usize,
            );
            let bl = (
                (point_rel_src.y * fheight).ceil() as usize,
                (point_rel_src.x * fwidth) as usize,
            );
            let br = (
                (point_rel_src.y * fheight).ceil() as usize,
                (point_rel_src.x * fwidth).ceil() as usize,
            );

            let color = interpolate(
                src[tl.0 * width + tl.1],
                src[tr.0 * width + tr.1],
                src[bl.0 * width + bl.1],
                src[br.0 * width + br.1],
                (
                    point_rel_src.y - point_rel_src.y.floor(),
                    point_rel_src.x - point_rel_src.x.floor(),
                ),
            );

            dest[dest_row * width + dest_col] = color;
        }
    }
}

#[derive(Copy, Clone)]
struct BufView(*mut MaybePixel);

unsafe impl Send for BufView {}
unsafe impl Sync for BufView {}

/// Split up a 2d plane into side*side chunks and sort them
/// by distance from the center, the idea is to redraw the
/// canvas in a circle emanating from the center to prioritize
/// the most interesting parts
fn chunks_2d(
    width: usize,
    height: usize,
    center: Point2D<usize, Pixels>,
    side: usize,
) -> Vec<(Range<usize>, Range<usize>)> {
    let mut chunks: Vec<_> = (0..width)
        .step_by(side)
        .flat_map(move |x_start| {
            (0..height).step_by(side).map(move |y_start| {
                (
                    x_start..min(x_start + side, width),
                    y_start..min(y_start + side, height),
                )
            })
        })
        .collect();

    chunks.sort_unstable_by_key(|(range_x, range_y)| {
        let x_diff = range_x.start.abs_diff(center.x as usize);
        let y_diff = range_y.start.abs_diff(center.y as usize);
        ((x_diff * x_diff) as f64 + (y_diff * y_diff) as f64).sqrt() as usize
    });
    chunks
}

#[inline]
fn do_one_pixel(
    c: usize,
    r: usize,
    width: usize,
    view: View,
    origin: Origin,
    iterations: usize,
    buf_view: BufView,
    ref_orbit: &RwLock<RefOrbit>,
) -> Option<NonZeroUsize> {
    let c_typed = Length::<_, Pixels>::new(c as f64);
    let r_typed = Length::<_, Pixels>::new(r as f64);

    let delta_x = c_typed * view;
    let delta_y = r_typed * view;
    // let delta_x = c as f64 / w as f64 * x_range;
    // let delta_y = r as f64 / h as f64 * y_range;

    let val;

    // if use_deltas {
    let lock = ref_orbit.read();
    val = if let Ok(val) = check_divergence_delta(
        &lock.orbit,
        &lock.orbit_f64,
        Complex {
            re: (delta_x - lock.delta_corr_x).0,
            im: (delta_y - lock.delta_corr_y).0,
        },
    ) {
        val
    } else {
        drop(lock); // the next line is not enough by itself to drop the read lock
        let mut lock = ref_orbit.write();

        let delta_x_bf = Length::<_, Units>::new(BigFloat::from(delta_x.get()));
        let delta_y_bf = Length::<_, Units>::new(BigFloat::from(delta_y.get()));

        let new_orbit = calculate_orbit(
            Complex {
                re: (origin.x() + delta_x_bf).0,
                im: (origin.y() + delta_y_bf).0,
            },
            iterations,
        )
        .into_boxed_slice();

        let val = check_orbit(&new_orbit);

        lock.orbit_f64 = new_orbit
            .iter()
            .map(|c| Complex::new(c.re.to_f64(), c.im.to_f64()))
            .collect();
        lock.orbit = new_orbit;

        lock.delta_corr_x = delta_x;
        lock.delta_corr_y = delta_y;

        val
    };
    // } else {
    //     val = check_orbit(&calculate_orbit(
    //         Complex::new(x_min.to_f64() + delta_x, y_min.to_f64() + delta_y),
    //         ITERATIONS,
    //     ))
    // }

    // let val = calculate(
    //     Complex::new(
    //         (c_typed * view + origin.x()).0,
    //         (r_typed * view + origin.y()).0,
    //     ),
    //     iterations,
    // );

    // SAFETY: all writes are disjoint
    unsafe {
        buf_view
            .0
            .add(r * width + c)
            .write(render_pixel(val).into());
    }

    val
}

struct RefOrbit {
    delta_corr_x: Length<f64, Units>,
    delta_corr_y: Length<f64, Units>,
    orbit: Box<[Complex<BigFloat>]>,
    orbit_f64: Box<[Complex<f64>]>,
}

pub fn render(
    buf: &mut [MaybePixel],
    width: usize,
    height: usize,
    center: Point2D<usize, Pixels>,
    coords: CoordinatesBox,
    iterations: usize,
    int: MultiInterrupter,
    tp: &ThreadPool,
) {
    let CoordinatesBox { origin, view } = coords;

    let buf_view = BufView(buf.as_mut_ptr());

    let ref_orbit = &{
        let orbit = calculate_orbit(
            Complex {
                re: coords.origin.x,
                im: coords.origin.y,
            },
            iterations,
        )
        .into_boxed_slice();

        let orbit_f64 = orbit
            .iter()
            .map(|c| Complex {
                re: c.re.to_f64(),
                im: c.im.to_f64(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        RwLock::new(RefOrbit {
            delta_corr_x: Length::new(0.0),
            delta_corr_y: Length::new(0.0),
            orbit,
            orbit_f64,
        })
    };

    let side = 20;
    let chunks = chunks_2d(width, height, center, side);

    let pbar = &ProgressBar::new(chunks.len() as u64);
    let int = &int;

    let (chunks_tx, ref chunks_rx) = crossbeam_channel::unbounded();
    chunks
        .into_iter()
        .try_for_each(|c| chunks_tx.send(c))
        .expect("crossbeam channel failed");

    tp.scope(|s| {
        for _ in 0..tp.current_num_threads() {
            s.spawn(move |_| {
                while let Ok((x_range, y_range)) = chunks_rx.try_recv() {
                    if int.interrupted() {
                        pbar.abandon();
                        return;
                    }

                    let buf_view = buf_view;

                    iproduct!(y_range, x_range).for_each(|(r, c)| {
                        do_one_pixel(c, r, width, view, origin, iterations, buf_view, ref_orbit);
                    });

                    // TODO: reimplement
                    // do the outline first, if all cells are black (don't diverge), then
                    // assume the whole block is black

                    // let x_inner_range = x_range.start + 1..x_range.end - 1;
                    // let y_inner_range = y_range.start + 1..y_range.end - 1;

                    // let outer_perimeter = x_range
                    //     .clone()
                    //     .map(|c| (c, y_range.start)) // top edge
                    //     .chain(x_range.clone().map(|c| (c, y_range.end - 1))) // bottom edge
                    //     .chain(y_inner_range.clone().map(|r| (x_range.start, r))) // left edge excluding top and bottom pixel
                    //     .chain(y_inner_range.clone().map(|r| (x_range.end - 1, r))); // right edge excluding top and bottom pixel

                    // let all_black = outer_perimeter.fold(true, |all_black, (c, r)| {
                    //     let val = do_one_pixel(c, r, width, view, origin, iterations, buf_view);
                    //     all_black && val.is_none()
                    // });

                    // if all_black {
                    //     iproduct!(y_inner_range, x_inner_range).for_each(|(r, c)| {
                    //         // SAFETY: all writes are disjoint
                    //         unsafe {
                    //             buf_view.0.add(r * width + c).write(0.into());
                    //         }
                    //     });
                    // } else {
                    //     iproduct!(y_inner_range, x_inner_range).for_each(|(r, c)| {
                    //         do_one_pixel(c, r, width, view, origin, iterations, buf_view);
                    //     });
                    // }

                    pbar.inc(1);
                }
            });
        }
    });

    pbar.finish();
}
