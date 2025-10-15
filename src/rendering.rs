use crate::drawing::maybe_pixel::MaybePixel;
use euclid::{Length, Point2D, Scale};
use gat_lending_iterator::LendingIterator;
use hsl::HSL;
use indicatif::ProgressBar;
use num::{Complex, Integer};
use palette::rgb::Rgb;
use palette::{Mix, Srgb, rgb};
use rayon::ThreadPool;
use simd_quad::*;
use std::cmp::min;
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

fn calculate(c: Complex<f64>, iterations: usize) -> f64 {
    let mut z: Complex<f64> = Complex::ZERO;
    for i in 0..iterations {
        z = z * z + c;
        if z.norm() > 4.0 {
            return i as f64 / iterations as f64;
        }
    }
    0.0
}

fn calculate_abs(c: Complex<f32>, iterations: usize) -> usize {
    let mut z: Complex<f32> = Complex::ZERO;
    for i in 0..iterations {
        z = z * z + c;
        if z.norm() > 4.0 {
            return i;
        }
    }
    0
}

fn calculate_simd<F>(c_re: X4<F>, c_im: X4<F>, iterations: usize) -> X4<<F as SimdX4>::BoolType>
where
    X4<F>: SimdX4Float<F>,
{
    #[allow(type_alias_bounds)]
    type U<T: SimdX4> = T::BoolType;

    let mut z_re = X4::<F>::dup(0.0);
    let mut z_im = X4::<F>::dup(0.0);

    let one_uint = X4::<U<F>>::dup(1u8);
    let two_float = X4::<F>::dup(2.0);
    let four_float = X4::<F>::dup(4.0);

    let mut final_i = X4::<U<F>>::dup(0u8);

    let mut z_re_sq = z_re * z_re;
    let mut z_im_sq = z_im * z_im;

    let mut i = X4::<U<F>>::dup(0u8);
    for _ in 0..iterations {
        let final_unset = final_i.eq_zero();

        // if all final_i are set
        if final_unset.all_zero() {
            return final_i;
        }

        // z = z * z + c;
        let new_z_re = z_re_sq - z_im_sq + c_re;
        z_im = two_float * z_re * z_im + c_im;
        z_re = new_z_re;

        z_re_sq = z_re * z_re;
        z_im_sq = z_im * z_im;
        let norm = (z_re_sq + z_im_sq).sqrt();
        let gt4 = norm.gt(four_float);

        final_i |= i & final_unset & gt4;

        i += one_uint;
    }

    final_i
}

struct RangeIterator<'a> {
    range_re: Range<usize>,
    im: f32,
    quad_im: X4<f32>,
    view: View,
    origin: Origin,
    iterations: usize,
    out: &'a mut [u32; u32::LANES],
}

impl<'a> RangeIterator<'a> {
    fn new<'b: 'a>(
        range_re: Range<usize>,
        im: f32,
        view: View,
        origin: Origin,
        iterations: usize,
        out: &'b mut [u32; u32::LANES],
    ) -> Self {
        assert_eq!(out.len(), f32::LANES);

        Self {
            range_re,
            im,
            quad_im: X4::dup(im),
            view,
            origin,
            iterations,
            out,
        }
    }
}

impl<'a> LendingIterator for RangeIterator<'a> {
    type Item<'b>
        = &'b [u32]
    where
        Self: 'b;

    fn next(&mut self) -> Option<Self::Item<'_>> {
        let range_len = self.range_re.len();
        if range_len >= f32::LANES {
            let mut chunk = [0.0; f32::LANES];
            for i in 0..f32::LANES {
                // SAFETY: we just checked
                let x = unsafe { self.range_re.next().unwrap_unchecked() };
                let re = Length::<_, Pixels>::new(x as f64) * self.view + self.origin.x();
                chunk[i] = re.0 as f32;
            }

            let quad_re = unsafe { X4::read_from_ptr(&chunk[0]) };
            let quad_result = calculate_simd(quad_re, self.quad_im, self.iterations);
            unsafe { quad_result.write_to_ptr(&raw mut self.out[0]) };
            Some(self.out)
        } else {
            for (i, x) in self.range_re.clone().enumerate() {
                let re = Length::<_, Pixels>::new(x as f64) * self.view + self.origin.x();
                self.out[i] = calculate_abs(
                    Complex {
                        re: re.0 as f32,
                        im: self.im,
                    },
                    self.iterations,
                ) as u32;
            }
            for i in range_len..self.out.len() {
                self.out[i] = u32::MAX; // sentinel value
            }
            Some(self.out)
        }
    }
}

fn _apply_sharpness(val: f64, sharpness: f64) -> f64 {
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

fn render_pixel(val: f64) -> u32 {
    if val == 0.0 {
        0
    } else {
        // let h = apply_sharpness(val, 30.0) / 6.0;
        // let l = apply_sharpness(val, 20.0) * 0.6;
        //
        // let hsl = HSL {
        //     h: h * 360.0,
        //     s: 1.0,
        //     l,
        // };

        let hsl = HSL {
            h: val * 360.0,
            s: 1.0,
            l: 0.5,
        };

        let (r, g, b) = hsl.to_rgb();

        rgb_to_u32(r, g, b)
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

fn chunks_2d(
    // canvas width
    width: usize,
    // canvas height
    height: usize,
    // chunk side length
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
        let x_diff = range_x.start.abs_diff(width / 2);
        let y_diff = range_y.start.abs_diff(height / 2);
        ((x_diff * x_diff) as f64 + (y_diff * y_diff) as f64).sqrt() as usize
    });
    chunks
}

pub fn render(
    buf: &mut [MaybePixel],
    width: usize,
    height: usize,
    coords: CoordinatesBox,
    iterations: usize,
    int: MultiInterrupter,
    tp: &ThreadPool,
) {
    let CoordinatesBox { origin, view } = coords;

    let buf_view = BufView(buf.as_mut_ptr());

    // side should be a multiple of 4 to optimize for simd
    let side = 20;
    let chunks = chunks_2d(width, height, side);

    let pbar = &ProgressBar::new(chunks.len() as u64);
    let int = &int;

    tp.scope(|s| {
        for (x_range, y_range) in chunks {
            s.spawn(move |_| {
                if int.interrupted() {
                    pbar.abandon();
                    return;
                }

                let buf_view = buf_view;

                let mut out_buf = [0; <f32 as simd_quad::SimdX4>::BoolType::LANES];

                // c ~ re
                // r ~ im

                for r in y_range {
                    let im = Length::<_, Pixels>::new(r as f64) * view + origin.y();

                    let mut iter = RangeIterator::new(
                        x_range.clone(),
                        im.0 as f32,
                        view,
                        origin,
                        iterations,
                        &mut out_buf,
                    );

                    iter.enumerate().for_each(|(i, out)| {
                        for (j, &v) in out.iter().enumerate() {
                            if v == u32::MAX {
                                break;
                            }

                            // SAFETY: all writes are disjoint
                            unsafe {
                                buf_view
                                    .0
                                    .add(r * width + (i * 4 + j))
                                    .write(render_pixel(v as f64 / iterations as f64).into());
                            }
                        }
                    });

                    // for c in x_range.clone() {
                    //     let c_typed = Length::<_, Pixels>::new(c as f64);
                    //     let r_typed = Length::<_, Pixels>::new(r as f64);

                    //     let val = calculate(
                    //         Complex::new(
                    //             (c_typed * view + origin.x()).0,
                    //             (r_typed * view + origin.y()).0,
                    //         ),
                    //         iterations,
                    //     );

                    //     // SAFETY: all writes are disjoint
                    //     unsafe {
                    //         buf_view
                    //             .0
                    //             .add(r * width + c)
                    //             .write(render_pixel(val).into());
                    //     }
                    // }
                }
                pbar.inc(1);
            });
        }
    });

    pbar.finish();
}
