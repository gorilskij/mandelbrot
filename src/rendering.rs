use euclid::{Length, Point2D, Scale};
use hsl::HSL;
use indicatif::ProgressBar;
use num::Complex;
use palette::rgb::Rgb;
use palette::{Mix, Srgb, rgb};
use rayon::prelude::*;

const ITERATIONS: usize = 2000;

pub enum Units {}
pub enum Pixels {}
pub enum Relative {}
pub type Point<U> = Point2D<f64, U>;
pub type View = Scale<f64, Pixels, Units>;

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

fn calculate(c: Complex<f64>) -> f64 {
    let mut z: Complex<f64> = Complex::ZERO;
    for i in 0..ITERATIONS {
        z = z * z + c;
        if z.norm() > 4.0 {
            return i as f64 / ITERATIONS as f64;
        }
    }
    0.0
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
    src_origin: Point<Units>,
    src_view: View,
    //
    dest_origin: Point<Units>,
    dest_view: View,
) {
    let fwidth = width as f64;
    let fheight = height as f64;

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
struct BufView(*mut u32);

unsafe impl Send for BufView {}
unsafe impl Sync for BufView {}

pub fn render(buf: &mut [u32], width: usize, height: usize, origin: Point<Units>, view: View) {
    let buf_view = BufView(buf.as_mut_ptr());

    let pbar = &ProgressBar::new(height as u64);
    (0..height).into_par_iter().for_each(move |r| {
        let buf_view = buf_view;

        for c in 0..width {
            let c_typed = Length::<_, Pixels>::new(c as f64);
            let r_typed = Length::<_, Pixels>::new(r as f64);

            let val = calculate(Complex::new(
                (c_typed * view + origin.x()).0,
                (r_typed * view + origin.y()).0,
            ));

            // SAFETY: all writes are disjoint
            unsafe {
                buf_view
                    .0
                    .add(r * width + c)
                    .write(render_pixel(val));
            }
        }
        pbar.inc(1);
    });
    pbar.finish();
}
