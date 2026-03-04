use crate::drawing::maybe_pixel::MaybePixel;
use crate::support::append_only::List as AOList;
use crate::support::{Length, Point, Scale, ToFBig};
use dashu::float::{DBig, FBig};
use hsl::HSL;
use indicatif::ProgressBar;
use itertools::{Itertools, iproduct};
use num::{Complex, Zero};
use ordered_float::OrderedFloat;
use palette::rgb::Rgb;
use palette::{Mix, Srgb, rgb};
use rayon::ThreadPool;
use std::cmp::min;
use std::fmt::Display;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    let k = 100_000.0.to_fbig();
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
        if x.re * x.re + x.im * x.im > 4.0 {
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
        if x.re * x.re + x.im * x.im > 4.0 {
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

// debug
// fn __get_divergence_delta(ref_orbit_f64: &Orbit<f64>, delta: Complex<f64>) -> Vec<Complex<f64>> {
//     // ref_orbit is [x_0, x_1, ...]
//     // delta is delta_0
//     // delta_{n+1} = 2 x_n delta_n + delta_n^2 + delta_0

//     let delta_0 = delta;
//     let mut delta = delta;
//     let mut out = vec![];
//     for (_, &c) in ref_orbit_f64.orbit.iter().enumerate() {
//         let x = c + delta;
//         out.push(x);
//         if x.re * x.re + x.im * x.im > 4.0 {
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
        // return rgb_to_u32(255, 255, 255);
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

        // let hsl = HSL {
        //     h: (val.get() as f64 / 1200.0).ln() % 1.0 * 360.0,
        //     s: 0.5,
        //     // l: 0.5,
        //     // l: ((val.get() as f64 / 40.0).sin() * 0.3 + 0.4),
        //     l: ((val.get() as f64 / 1.0).ln().sin() > 0.0) as u8 as f64,
        // };

        let (r, g, b) = hsl.to_rgb();
        rgb_to_u32(r, g, b)
    } else {
        0
    }
}

fn interpolate(
    color_tl: MaybePixel, // top left
    color_tr: MaybePixel, // top right
    color_bl: MaybePixel, // bottom left
    color_br: MaybePixel, // bottom right
    /* (y, x) where tl is (0, 0) */
    location: (f64, f64),
) -> MaybePixel {
    let Some(color_tl) = color_tl.get() else {
        return MaybePixel::none();
    };
    let Some(color_tr) = color_tr.get() else {
        return MaybePixel::none();
    };
    let Some(color_bl) = color_bl.get() else {
        return MaybePixel::none();
    };
    let Some(color_br) = color_br.get() else {
        return MaybePixel::none();
    };

    let color_tl = Rgb::<Srgb, _>::from(color_tl).into_format();
    let color_tr = Rgb::<Srgb, _>::from(color_tr).into_format();
    let color_bl = Rgb::<Srgb, _>::from(color_bl).into_format();
    let color_br = Rgb::<Srgb, _>::from(color_br).into_format();

    let top = color_tl.mix(color_tr, location.1);
    let bottom = color_bl.mix(color_br, location.1);

    let color = top.mix(bottom, location.0).into_format();

    color.into_u32::<rgb::channels::Argb>().into()
}

pub fn sample_zoomed(
    src: &[MaybePixel],
    dest: &mut [MaybePixel],
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
    let dest_origin_rel_src = {
        let p = &(dest_origin - src_origin).cast(|f| f.to_f64().value()) / src_view;
        Point::<_, Relative>::new(p.x, p.y)
    };

    for dest_row in 0..height {
        for dest_col in 0..width {
            let point_rel_dest = Point::<_, Relative>::new(dest_col as f64, dest_row as f64);

            let point_rel_src = (dest_origin_rel_src + point_rel_dest * (*dest_view / *src_view))
                .clamp(&Point::zero(), &Point::new(fwidth - 1.0, fheight - 1.0));

            let x_floor = point_rel_src.x as usize;
            let x_ceil = point_rel_src.x.ceil() as usize;
            let y_floor = point_rel_src.y as usize;
            let y_ceil = point_rel_src.y.ceil() as usize;

            let color = interpolate(
                src[y_floor * width + x_floor], // top-left
                src[y_floor * width + x_ceil],  // top-right
                src[y_ceil * width + x_floor],  // bottom-left
                src[y_ceil * width + x_ceil],   // bottom-right
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

const CHUNK_SIDE_POW: usize = 6; // chunk side length = 2^CHUNK_SIDE_POW

/// Split up a 2d plane into side*side chunks and sort them
/// by distance from the center, the idea is to redraw the
/// canvas in a circle emanating from the center or cursor
/// to prioritize the most interesting parts
fn chunks_2d(
    width: usize,
    height: usize,
    center: Point<usize, Pixels>,
) -> Vec<(Range<usize>, Range<usize>)> {
    let side = 1 << CHUNK_SIDE_POW;

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
        let x_diff = range_x.start.abs_diff(center.x) as f64;
        let y_diff = range_y.start.abs_diff(center.y) as f64;
        OrderedFloat(x_diff * x_diff + y_diff * y_diff)
    });
    chunks
}

fn chunk_order(x_range: Range<usize>, y_range: Range<usize>) -> Vec<(usize, usize)> {
    let mut vec = Vec::with_capacity(x_range.len() * y_range.len());

    let local_x_range = x_range.clone().step_by(1 << (CHUNK_SIDE_POW - 1));
    let local_y_range = y_range.clone().step_by(1 << (CHUNK_SIDE_POW - 1));
    vec.extend(iproduct!(local_x_range, local_y_range));

    for side_pow in (1..CHUNK_SIDE_POW).rev() {
        let local_x_range = (0..x_range.len()).step_by(1 << (side_pow - 1));
        let local_y_range = (0..y_range.len()).step_by(1 << (side_pow - 1));

        vec.extend(
            iproduct!(local_x_range, local_y_range)
                .filter(|(x, y)| x % (1 << side_pow) != 0 || y % (1 << side_pow) != 0)
                .map(|(x, y)| (x + x_range.start, y + y_range.start)),
        );
    }
    vec
}

// calculate divergence time from the delta of a reference orbit, if the
// reference orbit diverges too far before the orbit being calculated
// (making it impossible to compute usable delta values), set the current
// point as the new reference point and replace the reference orbit with
// the current point's orbit (affects all future delta calculations)
fn render_pixel(
    c: usize,
    r: usize,
    width: usize,
    view: View,
    origin: &Origin,
    iterations: usize,
    buf_view: BufView,
    ref_orbits: &AOList<RefOrbit>,
) -> Option<NonZeroUsize> {
    let c_typed = Length::<_, Pixels>::new(c as f64);
    let r_typed = Length::<_, Pixels>::new(r as f64);

    let delta = Complex {
        re: (c_typed * view).inner,
        im: (r_typed * view).inner,
    };

    let val = ref_orbits
        .iter()
        .find_map(|ref_orbit| {
            let out = check_divergence_delta(&ref_orbit.orbit, delta - ref_orbit.delta_corr).ok();

            #[cfg(debug_assertions)]
            if out.is_some() {
                ref_orbit.hits.fetch_add(1, Ordering::Relaxed);
            }

            out
        })
        .unwrap_or_else(|| {
            let delta_fbig = Complex {
                re: delta.re.to_fbig(),
                im: delta.im.to_fbig(),
            };

            let (_, new_orbit) = calculate_orbit(
                Complex {
                    re: origin.x.clone(),
                    im: origin.y.clone(),
                } + delta_fbig,
                iterations,
            );

            // guaranteed to be Ok(_)
            let val = check_orbit(&new_orbit).unwrap();

            ref_orbits.push_front(RefOrbit {
                delta_corr: delta,
                orbit: new_orbit,
                #[cfg(debug_assertions)]
                hits: Default::default(),
            });

            val
        });

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
    orbit: Orbit<f64>,
    #[cfg(debug_assertions)]
    hits: AtomicUsize,
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

    let ref_orbits = &{
        let (_, orbit) = calculate_orbit(
            Complex {
                re: origin.x.clone(),
                im: origin.y.clone(),
            },
            iterations,
        );

        let list = AOList::new();
        list.push_front(RefOrbit {
            delta_corr: Complex::ZERO,
            orbit,
            #[cfg(debug_assertions)]
            hits: Default::default(),
        });
        list
    };

    let chunks = chunks_2d(width, height, center);

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

                    // render the top-left pixel
                    let first_pixel = render_pixel(
                        x_range.start,
                        y_range.start,
                        width,
                        view,
                        origin,
                        iterations,
                        buf_view,
                        ref_orbits,
                    );

                    // if the top-left pixel is black, check if the whole perimeter of the chunk is black
                    if first_pixel.is_none() {
                        let outer_perimeter = (x_range.start + 1..x_range.end)
                            .map(|c| (c, y_range.start)) // top edge excluding the first pixel
                            .chain(x_range.clone().map(|c| (c, y_range.end - 1))) // bottom edge
                            .chain(y_inner_range.clone().map(|r| (x_range.start, r))) // left edge excluding top and bottom pixel
                            .chain(y_inner_range.clone().map(|r| (x_range.end - 1, r))); // right edge excluding top and bottom pixel

                        let all_black = outer_perimeter.fold(true, |all_black, (c, r)| {
                            let val = render_pixel(
                                c, r, width, view, origin, iterations, buf_view, ref_orbits,
                            );
                            all_black && val.is_none()
                        });

                        // if the perimeter is all black, assume the rest of the chunk is black
                        if all_black {
                            iproduct!(x_inner_range, y_inner_range).for_each(|(c, r)| {
                                // SAFETY: all writes are disjoint
                                unsafe {
                                    buf_view.0.add(r * width + c).write(0.into());
                                    // debug
                                    // buf_view
                                    //     .0
                                    //     .add(r * width + c)
                                    //     .write(rgb_to_u32(255, 255, 255).into());
                                }
                            });
                        }
                        // if the perimeter is not all-black, render the rest of the chunk
                        else {
                            let out = chunk_order(x_inner_range.clone(), y_inner_range.clone());
                            assert_eq!(
                                out.len(),
                                x_inner_range.len() * y_inner_range.len(),
                                "{:?}\n{:?}",
                                x_inner_range,
                                y_inner_range
                            );

                            out.into_iter().for_each(|(c, r)| {
                                render_pixel(
                                    c, r, width, view, origin, iterations, buf_view, ref_orbits,
                                );
                            });

                            // DEBUG
                            // chunk_order(x_range, y_range)
                            //     .into_iter()
                            //     .for_each(|(c, r)| {
                            //         render_pixel(
                            //             c, r, width, view, origin, iterations, buf_view, ref_orbits,
                            //         );
                            //     });
                        }
                    }
                    // if the top-left pixel is not black, render the rest of the chunk
                    else {
                        chunk_order(x_range, y_range)
                            .into_iter()
                            .skip(1)
                            .for_each(|(c, r)| {
                                render_pixel(
                                    c, r, width, view, origin, iterations, buf_view, ref_orbits,
                                );
                            });
                    }

                    pbar.inc(1);
                }
            });
        }
    });

    pbar.finish();

    #[cfg(debug_assertions)]
    {
        println!("{:>10} {:>10}", "length", "hits");
        for o in ref_orbits.iter() {
            println!(
                "{:>10} {:>10}",
                o.orbit.orbit.len(),
                o.hits.load(Ordering::Relaxed)
            );
        }
    }
}
