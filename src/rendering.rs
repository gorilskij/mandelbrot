use crate::drawing::maybe_pixel::MaybePixel;
use crate::support::{Length, Point, Scale, ToFBig};
use cpu_time::ProcessTime;
use dashu::float::FBig;
use hsl::HSL;
use indicatif::ProgressBar;
use itertools::{Itertools, iproduct};
use num::{Complex, Zero};
use palette::rgb::Rgb;
use palette::{Mix, Srgb, rgb};
use parking_lot::RwLock;
use rand::{Rng, rng};
use rayon::ThreadPool;
use std::assert_matches::assert_matches;
use std::cmp::min;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use waker_interrupter::MultiInterrupter;

#[derive(Copy, Clone)]
pub enum Units {}
#[derive(Copy, Clone)]
pub enum Pixels {}
#[derive(Copy, Clone)]
pub enum Relative {}
pub type Origin = Point<FBig, Units>;
pub type View = Scale<f64, Pixels, Units>;

#[derive(Clone)]
pub struct CoordinatesBox {
    pub origin: Origin,
    pub view: View,
}

// trait TypedXY<T, U> {
//     fn x(&self) -> Length<T, U>;
//     fn y(&self) -> Length<T, U>;
// }

// impl<T, U> TypedXY<T, U> for Point2D<T, U>
// where
//     T: Copy,
// {
//     fn x(&self) -> Length<T, U> {
//         Length::new(self.x)
//     }

//     fn y(&self) -> Length<T, U> {
//         Length::new(self.y)
//     }
// }

fn is_bad_value(v: &Complex<FBig>) -> bool {
    // implicitly also checks that neither is NaN
    let k = 1000.0.to_fbig();
    !(-&k <= v.re && v.re <= k && -&k <= v.im && v.im <= k)
}

struct Orbit<F> {
    is_full: bool,
    orbit: Box<[Complex<F>]>,
}

fn calculate_orbit(x_0: Complex<FBig>, iterations: usize) -> (Orbit<FBig>, Orbit<f64>) {
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
    let out_f64 = out
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
            orbit: out_f64,
        },
    )
}

fn check_orbit(orbit: &Orbit<f64>) -> Result<Option<NonZeroUsize>, ()> {
    for (i, x) in orbit.orbit.iter().enumerate() {
        if x.norm() > 4.0 {
            // this is always Some(...), i + 1 can't be 0
            return Ok(NonZeroUsize::new(i + 1));
        }
    }
    if orbit.is_full { Ok(None) } else { Err(()) }
}

fn check_divergence_delta(
    ref_orbit_f64: &Orbit<f64>,
    delta: Complex<f64>,
) -> Result<Option<NonZeroUsize>, ()> {
    // ref_orbit is [x_0, x_1, ...]
    // delta is delta_0
    // delta_{n+1} = 2 x_n delta_n + delta_n^2 + delta_0

    let delta_0 = delta;
    let mut delta = delta;
    for (i, &c) in ref_orbit_f64.orbit.iter().enumerate() {
        let x = c + delta;
        if x.norm() > 4.0 {
            // this is always Some(...), i + 1 can't be 0
            return Ok(NonZeroUsize::new(i + 1));
        }

        delta = c * 2.0 * delta + delta * delta + delta_0;
    }

    if ref_orbit_f64.is_full {
        Ok(None)
    } else {
        Err(())
    }
}

// fn __get_divergence_delta(ref_orbit_f64: &Orbit<f64>, delta: Complex<f64>) -> Vec<Complex<f64>> {
//     // ref_orbit is [x_0, x_1, ...]
//     // delta is delta_0
//     // delta_{n+1} = 2 x_n delta_n + delta_n^2 + delta_0

//     let delta_0 = delta;
//     let mut delta = delta;
//     let mut out = vec![];
//     for (i, &c) in ref_orbit_f64.orbit.iter().enumerate() {
//         let x = c + delta;
//         out.push(x);
//         if x.norm() > 4.0 {
//             // this is always Some(...), i + 1 can't be 0
//             return out;
//         }

//         delta = c * 2.0 * delta + delta * delta + delta_0;
//     }

//     out
// }

fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

fn val_to_color(val: Option<NonZeroUsize>) -> u32 {
    if let Some(val) = val {
        // [0, 1)
        // let f = 1.0 - 1.0 / (val.get() as f64 / 100.0 + 1.0);

        // let hsl = HSL {
        //     // h: val.get() as f64 % 360.0,
        //     // s: 0.7,
        //     // // l: 0.5,
        //     // l: (val.get() as f64 / 10.0).sin() * 0.1 + 0.5,
        //     h: (val.get() as f64 / 10.0).sin() * 180.0,
        //     s: 0.7,
        //     // l: 0.5,
        //     l: ((val.get() as f64 / (10.0 * std::f64::consts::E)).sin() * 0.3 + 0.4),
        // };

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
    src_coords: &CoordinatesBox,
    dest_coords: &CoordinatesBox,
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
        let p = &(dest_origin - src_origin).cast(|f| f.to_f64().value()) / src_view;
        Point::<_, Relative>::new(p.x / fwidth, p.y / fheight)
    };

    for dest_row in 0..height {
        for dest_col in 0..width {
            // [0, 1] coordinates relative to the reference frame given by dest
            let point_rel_dest =
                Point::<_, Relative>::new(dest_col as f64 / fwidth, dest_row as f64 / fheight);

            // calculate [0, 1] coordinates in the reference frame given by src
            let point_rel_src = (dest_origin_rel_src + point_rel_dest * (*dest_view / *src_view))
                .clamp(
                    &Point::zero(),
                    &Point::new((fwidth - 1.0) / fwidth, (fheight - 1.0) / fheight),
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
    center: Point<usize, Pixels>,
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
        let x_diff = range_x.start.abs_diff(center.x);
        let y_diff = range_y.start.abs_diff(center.y);
        (x_diff as f64).hypot(y_diff as f64) as usize
    });
    chunks
}

// calculate divergence time from the delta of a reference orbit, if the
// reference orbit diverges too far before the orbit being calculated
// (making it impossible to compute usable delta values), set the current
// point as the new reference point and replace the reference orbit with
// the current point's orbit (affects all future delta calculations)
#[inline]
fn render_pixel(
    c: usize,
    r: usize,
    width: usize,
    view: View,
    origin: &Origin,
    iterations: usize,
    buf_view: BufView,
    ref_orbit: &RwLock<RefOrbit>,
    writer_active: &AtomicBool,
) -> Option<NonZeroUsize> {
    let c_typed = Length::<_, Pixels>::new(c as f64);
    let r_typed = Length::<_, Pixels>::new(r as f64);

    let delta = Complex {
        re: (c_typed * view).inner,
        im: (r_typed * view).inner,
    };

    let val = loop {
        // this block is necessary to properly drop the read lock
        // let __len;
        {
            let lock = ref_orbit.read();
            // __len = lock.orbit.orbit.len();
            if let Ok(val) = check_divergence_delta(&lock.orbit_f64, delta - lock.delta_corr) {
                break val;
            } else {
                assert_matches!(
                    check_divergence_delta(&lock.orbit_f64, delta - lock.delta_corr),
                    Err(_)
                );
            }
        }

        if writer_active
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // the next iteration will naturally block until the writer is done
            continue;
        }
        let mut lock = ref_orbit.write();

        // let len__ = lock.orbit.orbit.len();
        // println!(
        //     "{} =? {} {}",
        //     __len,
        //     len__,
        //     if __len == len__ { "" } else { "ERROR" }
        // );

        let recalc_id = rng().random::<u8>();
        println!("({recalc_id}) recalculating reference orbit");

        let delta_bf = Complex {
            re: delta.re.to_fbig(),
            im: delta.im.to_fbig(),
        };

        let start = ProcessTime::now();
        let (new_orbit, new_orbit_f64) = calculate_orbit(
            Complex {
                re: origin.x.clone(),
                im: origin.y.clone(),
            } + delta_bf,
            iterations,
        );
        println!("({recalc_id}) done {:?}", start.elapsed());

        // guaranteed to be Ok(_)
        let val = check_orbit(&new_orbit_f64).unwrap();

        println!("{} -> {}", lock.orbit.orbit.len(), new_orbit.orbit.len());
        // if new_orbit_f64.orbit.len() < lock.orbit_f64.orbit.len() {
        //     println!("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
        // }
        // assert_matches!(
        //     check_divergence_delta(&lock.orbit_f64, delta - lock.delta_corr),
        //     Err(_)
        // );
        // assert_matches!(check_divergence_delta(&new_orbit_f64, Complex::ZERO), Ok(_));
        // if new_orbit_f64.orbit.len() < lock.orbit_f64.orbit.len() {
        //     let delta_orb = __get_divergence_delta(&lock.orbit_f64, delta - lock.delta_corr);
        //     for (a, b) in delta_orb.iter().zip(&new_orbit_f64.orbit) {
        //         println!(">>   {a:>20?} {b:>20?})");
        //     }
        // }

        lock.orbit = new_orbit;
        lock.orbit_f64 = new_orbit_f64;

        lock.delta_corr = delta;

        drop(lock);
        writer_active.store(false, Ordering::Release);

        break val;
    };

    // SAFETY: all writes are disjoint
    unsafe {
        buf_view
            .0
            .add(r * width + c)
            .write(val_to_color(val).into());
    }

    val
}

struct RefOrbit {
    delta_corr: Complex<f64>,
    orbit: Orbit<FBig>,
    orbit_f64: Orbit<f64>,
}

pub fn render(
    buf: &mut [MaybePixel],
    width: usize,
    height: usize,
    center: Point<usize, Pixels>,
    coords: CoordinatesBox,
    iterations: usize,
    int: MultiInterrupter,
    tp: &ThreadPool,
) {
    let CoordinatesBox { ref origin, view } = coords;

    let buf_view = BufView(buf.as_mut_ptr());

    let ref_orbit = &{
        let (orbit, orbit_f64) = calculate_orbit(
            Complex {
                re: origin.x.clone(),
                im: origin.y.clone(),
            },
            iterations,
        );

        RwLock::new(RefOrbit {
            delta_corr: Complex::ZERO,
            orbit,
            orbit_f64,
        })
    };
    let writer_active = &AtomicBool::new(false);

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

                    // do the outline first, if all cells are black (don't diverge), then
                    // assume the whole block is black

                    let x_inner_range = x_range.start + 1..x_range.end - 1;
                    let y_inner_range = y_range.start + 1..y_range.end - 1;

                    let outer_perimeter = x_range
                        .clone()
                        .map(|c| (c, y_range.start)) // top edge
                        .chain(x_range.clone().map(|c| (c, y_range.end - 1))) // bottom edge
                        .chain(y_inner_range.clone().map(|r| (x_range.start, r))) // left edge excluding top and bottom pixel
                        .chain(y_inner_range.clone().map(|r| (x_range.end - 1, r))); // right edge excluding top and bottom pixel

                    let all_black = outer_perimeter.fold(true, |all_black, (c, r)| {
                        let val = render_pixel(
                            c,
                            r,
                            width,
                            view,
                            origin,
                            iterations,
                            buf_view,
                            ref_orbit,
                            writer_active,
                        );
                        all_black && val.is_none()
                    });

                    let mut inner_pixels = iproduct!(x_inner_range, y_inner_range).collect_vec();
                    inner_pixels.sort_unstable_by_key(|(c, r)| {
                        let x_diff = c.abs_diff(center.x);
                        let y_diff = r.abs_diff(center.y);
                        (x_diff as f64).hypot(y_diff as f64) as usize
                    });

                    if all_black {
                        inner_pixels.into_iter().for_each(|(c, r)| {
                            // SAFETY: all writes are disjoint
                            unsafe {
                                buf_view.0.add(r * width + c).write(0.into());
                            }
                        });
                    } else {
                        inner_pixels.into_iter().for_each(|(c, r)| {
                            render_pixel(
                                c,
                                r,
                                width,
                                view,
                                origin,
                                iterations,
                                buf_view,
                                ref_orbit,
                                writer_active,
                            );
                        });
                    }

                    pbar.inc(1);
                }
            });
        }
    });

    pbar.finish();
}
