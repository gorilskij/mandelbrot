mod image_buffer;

use num::complex::Complex;
use minifb::{Key, MouseMode, Window, WindowOptions};
use indicatif::ProgressBar;
use rayon::prelude::*;
use hsl::HSL;

const FILE_PATH: &str = "test.png";
const ITERATIONS: usize = 2000;
const OVERSAMPLE: u32 = 2;

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
        // -1.0 / (1.0 - 1.0 / sharpness) * (val - 1.0)
        1.0
    }
}

fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

fn render_pixel(val: f64) -> u32 {
    if val == 0.0 {
        0
    } else {
        let h = apply_sharpness(val, 30.0) / 6.0;
        let l = apply_sharpness(val, 20.0) * 0.6;

        let hsl = HSL {
            h: h * 360.0,
            s: 1.0,
            l,
        };

        let (r, g, b) = hsl.to_rgb();

        rgb_to_u32(r, g, b)
    }
}

fn render(
    buf: &mut [u32],
    w: usize,
    h: usize,
    // the coordinate of the top-left corner (x, y)
    origin: (f64, f64),
    scale: f64,
) {
    let (x_min, y_min) = origin;
    let x_range = w as f64 * scale;
    let y_range = h as f64 * scale;

    let pbar = ProgressBar::new(h as u64);
    (0..h).for_each(|r| {
        for c in 0..w {
            let x = c as f64 / w as f64 * x_range + x_min;
            let y = r as f64 / h as f64 * y_range + y_min;
            let val = calculate(Complex::new(x, y));
            // row_buf[c] = val;

            buf[r * w + c] = render_pixel(val);
        }
        pbar.inc(1);
    });
    pbar.finish();
}

fn main() {
    let w = 1000;
    let h = 600;

    let mut buffer: Vec<u32> = vec![0; w * h];

    let mut window = Window::new(
        "Test - ESC to exit",
        w, h,
        WindowOptions::default(),
    )
        .unwrap();

    window.set_target_fps(60);

    let mut origin = (-1.0, -1.0);
    let mut scale = 1.0 / 1000.0;

    let mut cached = None;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if let Some((mouse_x, mouse_y)) = window.get_mouse_pos(MouseMode::Discard) {
            if let Some((scroll_x, scroll_y)) = window.get_scroll_wheel() {
                scale *= 1.0 - (scroll_y as f64 / 100.0).clamp(-0.1, 0.1);
                println!("new scale {}", scale);
            }
        }

        let cache_key = (origin, scale);
        if cached != Some(cache_key) {
            render(&mut buffer, w, h, origin, scale);
            cached = Some(cache_key);
        }

        window
            .update_with_buffer(&buffer, w, h)
            .unwrap();
    }
}
