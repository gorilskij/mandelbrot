use hsl::HSL;
use indicatif::ProgressBar;
use minifb::{Key, KeyRepeat, MouseMode, Window, WindowOptions};
use num::complex::Complex;
use palette::rgb::Rgb;
use palette::{Mix, Srgb, rgb};
use rayon::prelude::*;

const ITERATIONS: usize = 2000;

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
    width: usize,
    height: usize,
    /* [0,1] relative to the source */
    center_x: f64,
    center_y: f64,
    multiplier: f64,
) {
    let fwidth = width as f64;
    let fheight = height as f64;

    for dest_row in 0..height {
        for dest_col in 0..width {
            let dest_y = dest_row as f64 / fheight;
            let dest_x = dest_col as f64 / fwidth;

            let src_y =
                ((dest_y - center_y) / multiplier + center_y).clamp(0.0, (fheight - 1.0) / fheight);
            let src_x =
                ((dest_x - center_x) / multiplier + center_x).clamp(0.0, (fwidth - 1.0) / fwidth);

            let tl = ((src_y * fheight) as usize, (src_x * fwidth) as usize);
            let tr = ((src_y * fheight) as usize, (src_x * fwidth).ceil() as usize);
            let bl = ((src_y * fheight).ceil() as usize, (src_x * fwidth) as usize);
            let br = (
                (src_y * fheight).ceil() as usize,
                (src_x * fwidth).ceil() as usize,
            );

            let color = interpolate(
                src[tl.0 * width + tl.1],
                src[tr.0 * width + tr.1],
                src[bl.0 * width + bl.1],
                src[br.0 * width + br.1],
                (src_y - src_y.floor(), src_x - src_x.floor()),
            );

            dest[dest_row * width + dest_col] = color;
        }
    }
}

#[derive(Copy, Clone)]
struct BufView(*mut u32);

unsafe impl Send for BufView {}
unsafe impl Sync for BufView {}

fn render(
    buf: &mut [u32],
    w: usize,
    h: usize,
    // the coordinate of the top-left corner (x, y)
    origin: (f64, f64),
    view: f64,
) {
    let (x_min, y_min) = origin;
    let x_range = w as f64 * view;
    let y_range = h as f64 * view;

    let buf_view = BufView(buf.as_mut_ptr());

    let pbar = &ProgressBar::new(h as u64);
    (0..h).into_par_iter().for_each(move |r| {
        let buf_view = buf_view;

        for c in 0..w {
            let x = c as f64 / w as f64 * x_range + x_min;
            let y = r as f64 / h as f64 * y_range + y_min;
            let val = calculate(Complex::new(x, y));

            // SAFETY: all writes are disjoint
            unsafe {
                buf_view
                    .0
                    .offset((r * w + c) as isize)
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
    multiplier: f64,
}

impl Buffer {
    fn new(width: usize, height: usize) -> Self {
        Self {
            base: vec![0; width * height].into_boxed_slice(),
            zoomed: vec![0; width * height].into_boxed_slice(),
            multiplier: 1.0,
        }
    }
}

fn main() {
    let w = 1000;
    let h = 600;

    let mut buffer = Buffer::new(w, h);

    let mut window = Window::new("Mandelbrot", w, h, WindowOptions::default()).unwrap();

    window.set_target_fps(60);

    let mut origin_x = -1.0;
    let mut origin_y = -1.0;
    // displayed range / pixel size (zooming in means reducing view)
    let mut view = 1.0 / 500.0;

    let mut cached = None;

    let mut first_time = true;

    while window.is_open() {
        let mut mouse01 = None;

        if let Some((mouse_x, mouse_y)) = window.get_mouse_pos(MouseMode::Discard) {
            if let Some((_, scroll_y)) = window.get_scroll_wheel() {
                let multiplier = 1.0 + (scroll_y as f64 / 100.0).clamp(-0.2, 0.2);

                mouse01 = Some((
                    mouse_x as f64 / w as f64,
                    mouse_y as f64 / h as f64,
                    multiplier,
                ));

                view /= multiplier;

                let pos_x = mouse_x as f64 * view + origin_x;
                let pos_y = mouse_y as f64 * view + origin_y;

                origin_x = pos_x + (origin_x - pos_x) / multiplier;
                origin_y = pos_y + (origin_y - pos_y) / multiplier;
            }
        }

        let cache_key = (origin_x, origin_y, view);
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            render(&mut buffer.base, w, h, (origin_x, origin_y), view);
            buffer.zoomed.copy_from_slice(&buffer.base);
            buffer.multiplier = 1.0;

            cached = Some(cache_key);
            // window.update_with_buffer(&buffer, w, h).unwrap();
        } else {
            if cached != Some(cache_key) {
                if first_time {
                    render(&mut buffer.base, w, h, (origin_x, origin_y), view);
                    buffer.zoomed.copy_from_slice(&buffer.base);
                    buffer.multiplier = 1.0;

                    first_time = false;
                } else if let Some((center_x, center_y, multiplier)) = mouse01 {
                    buffer.multiplier *= multiplier;
                    sample_zoomed(&buffer.base, &mut buffer.zoomed, w, h, center_x, center_y, buffer.multiplier);
                }
                cached = Some(cache_key);
            }
        }

        window.update_with_buffer(&buffer.zoomed, w, h).unwrap();
    }
}
