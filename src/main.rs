use euclid::{Point2D, UnknownUnit};
use hsl::HSL;
use indicatif::ProgressBar;
use minifb::{Key, KeyRepeat, MouseMode, Window, WindowOptions};
use num::complex::Complex;
use palette::rgb::Rgb;
use palette::{Mix, Srgb, rgb};
use rayon::prelude::*;

const ITERATIONS: usize = 2000;

type Point = Point2D<f64, UnknownUnit>;

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
    assert!(val >= 0.0 && val <= 1.0);
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

fn sample_zoomed(
    src: &[u32],
    dest: &mut [u32],
    // src and dest must match
    width: usize,
    height: usize,
    //
    src_origin: Point,
    src_view: f64,
    //
    dest_origin: Point,
    dest_view: f64,
) {
    let fwidth = width as f64;
    let fheight = height as f64;

    // src_origin and dest_origin are both absolute
    // calculate dest_origin in the [0, 1] reference frame given by src
    let dest_origin_rel_src = ((dest_origin - src_origin) / src_view).to_point();

    for dest_row in 0..height {
        for dest_col in 0..width {
            // [0, 1] coordinates relative to the reference frame given by dest
            let point_rel_dest = Point::new(dest_col as f64 / fwidth, dest_row as f64 / fheight);

            // calculate [0, 1] coordinates in the reference frame given by src
            let point_rel_src =
                (dest_origin_rel_src + point_rel_dest.to_vector() * (dest_view / src_view)).clamp(
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

fn render(buf: &mut [u32], width: usize, height: usize, origin: Point, view: f64) {
    let buf_view = BufView(buf.as_mut_ptr());

    let pbar = &ProgressBar::new(height as u64);
    (0..height).into_par_iter().for_each(move |r| {
        let buf_view = buf_view;

        for c in 0..width {
            let val = calculate(Complex::new(
                c as f64 * view + origin.x,
                r as f64 * view + origin.y,
            ));

            // SAFETY: all writes are disjoint
            unsafe {
                buf_view
                    .0
                    .offset((r * width + c) as isize)
                    .write(render_pixel(val));
            }
        }
        pbar.inc(1);
    });
    pbar.finish();
}

struct Buffer {
    base: Box<[u32]>,
    zoomed: Box<[u32]>,
    //
    base_origin: Point,
    base_view: f64,
}

impl Buffer {
    fn new(width: usize, height: usize, origin: Point, view: f64) -> Self {
        Self {
            base: vec![0; width * height].into_boxed_slice(),
            zoomed: vec![0; width * height].into_boxed_slice(),
            base_origin: origin,
            base_view: view,
        }
    }
}

fn main() {
    let width = 1000;
    let height = 600;

    let mut window = Window::new("Mandelbrot", width, height, WindowOptions::default()).unwrap();

    window.set_target_fps(60);

    let mut origin = Point::new(-2.5, -1.0);

    // Displayed range / pixel size (zooming in means reducing view)
    // (real world units / pixel)
    let mut view = 1.0 / 300.0;

    let mut buffer = Buffer::new(width, height, origin, view);

    let mut cached = None;

    let mut first_time = true;

    while window.is_open() {
        let mut zoomed = false;

        if let Some((mouse_x, mouse_y)) = window.get_mouse_pos(MouseMode::Discard) {
            if let Some((_, scroll_y)) = window.get_scroll_wheel() {
                zoomed = true;

                // let cursor_rel = Point::new(mouse_x as f64, mouse_y as f64);
                // let cursor_abs = origin + (cursor_rel * view).to_vector();

                // let pos_x = mouse_x as f64 * view + origin.x;
                // let pos_y = mouse_y as f64 * view + origin.y;

                // origin.x = pos.x + (origin.x - pos.x) / multiplier;
                // origin.y = pos.y + (origin.y - pos.y) / multiplier;

                // origin = cursor_abs + (origin - cursor_abs) / multiplier;


                // [0, 1] in the reference frame given by the window
                let cursor_rel = Point::new(
                    mouse_x as f64 / width as f64,
                    mouse_y as f64 / height as f64,
                );

                let cursor_abs = origin + (cursor_rel * view).to_vector();

                let multiplier = 1.0 + (scroll_y as f64 / 100.0).clamp(-0.2, 0.2);
                view /= multiplier;

                origin = cursor_abs + (origin - cursor_abs) / multiplier;

                println!("{:?} {}", origin, view);
            }
        }

        let cache_key = (origin, view);
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            render(&mut buffer.base, width, height, origin, view);
            buffer.zoomed.copy_from_slice(&buffer.base);
            buffer.base_origin = origin;
            buffer.base_view = view;

            cached = Some(cache_key);
            // window.update_with_buffer(&buffer, w, h).unwrap();
        } else {
            if cached != Some(cache_key) {
                if first_time {
                    render(&mut buffer.base, width, height, origin, view);
                    buffer.zoomed.copy_from_slice(&buffer.base);
                    buffer.base_origin = origin;
                    buffer.base_view = view;

                    first_time = false;
                } else if zoomed {
                    sample_zoomed(
                        &buffer.base,
                        &mut buffer.zoomed,
                        //
                        width,
                        height,
                        //
                        buffer.base_origin,
                        buffer.base_view,
                        //
                        origin,
                        view,
                    );
                }
                cached = Some(cache_key);
            }
        }

        window
            .update_with_buffer(&buffer.zoomed, width, height)
            .unwrap();
    }
}
