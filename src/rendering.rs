use crate::drawing::maybe_pixel::MaybePixel;
use crossbeam_channel;
use euclid::{Length, Point2D, Scale};
use hsl::HSL;
use indicatif::ProgressBar;
use num::Complex;
use palette::rgb::Rgb;
use palette::{Mix, Srgb, rgb};
use rayon::ThreadPool;
use std::cmp::min;
use std::num::NonZeroUsize;
use std::ops::Range;
use waker_interrupter::MultiInterrupter;

pub enum Units {}
pub enum Pixels {}
pub enum Relative {}
pub type Point<U> = Point2D<f64, U>;
pub type Origin = Point<Units>;
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

fn calculate(c: Complex<f64>, iterations: usize) -> Option<NonZeroUsize> {
    let mut z: Complex<f64> = Complex::ZERO;
    for i in 1..iterations + 1 {
        z = z * z + c;
        if z.norm() > 4.0 {
            // SAFETY: i starts iterating at 1, it is never 0
            unsafe { return Some(NonZeroUsize::new_unchecked(i)) };
        }
    }
    None
}

fn apply_sharpness(val: f64, sharpness: f64) -> f64 {
    assert!((0.0..=1.0).contains(&val));
    assert!(sharpness >= 2.0);
    if val < 1.0 / sharpness {
        sharpness * val
    } else {
        -1.0 / (1.0 - 1.0 / sharpness) * (val - 1.0)
        // 1.0
    }
}

fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

fn render_pixel(val: Option<NonZeroUsize>) -> u32 {
    if let Some(val) = val {
        // [0, 1)
        let f = 1.0 - 1.0 / (val.get() as f64 / 100.0 + 1.0);

        let hsl = HSL {
            // h: val.get() as f64 % 360.0,
            // s: 0.7,
            // // l: 0.5,
            // l: (val.get() as f64 / 10.0).sin() * 0.1 + 0.5,
            h: (val.get() as f64 / 10.0).sin() * 180.0,
            s: 0.7,
            // l: 0.5,
            l: (val.get() as f64 / (10.0 * std::f64::consts::E)).sin() * 0.4 + 0.5,
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
    let dest_origin_rel_src = {
        let p = ((dest_origin - src_origin) / src_view).to_point();
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

                    for r in y_range {
                        for c in x_range.clone() {
                            let c_typed = Length::<_, Pixels>::new(c as f64);
                            let r_typed = Length::<_, Pixels>::new(r as f64);

                            let val = calculate(
                                Complex::new(
                                    (c_typed * view + origin.x()).0,
                                    (r_typed * view + origin.y()).0,
                                ),
                                iterations,
                            );

                            // SAFETY: all writes are disjoint
                            unsafe {
                                buf_view
                                    .0
                                    .add(r * width + c)
                                    .write(render_pixel(val).into());
                            }
                        }
                    }
                    pbar.inc(1);
                }
            });
        }
    });

    pbar.finish();
}
