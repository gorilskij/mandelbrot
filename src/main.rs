#![feature(f16)]

mod complex;

use std::fmt::{Display, Formatter};
use complex::{Complex, Float, Zero};
use hsl::HSL;
use indicatif::ProgressBar;
use minifb::{Key, KeyRepeat, MouseMode, Window, WindowOptions};
use rayon::prelude::*;
use f256::f256;

const ITERATIONS: usize = 2000;

fn calculate<F: Float>(c: Complex<F>) -> F {
    let mut z = Complex::<F>::ZERO;
    for i in 0..ITERATIONS {
        z = z * z + c;
        if z.norm() > F::from_f64(4.0) {
            return F::from_usize(i) / F::from_usize(ITERATIONS);
        }
    }
    F::from_f64(0.0)
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

fn render_pixel<F: Float>(val: F) -> u32 {
    if val == F::from_f64(0.0) {
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
            h: (val * F::from_f64(360.0)).into_f64(),
            s: 1.0,
            l: 0.5,
        };

        let (r, g, b) = hsl.to_rgb();

        rgb_to_u32(r, g, b)
    }
}

// Returns (&source, &mut destination)
fn select_rows(
    buf: &mut [u32],
    width: usize,
    source_row: usize,
    destination_row: usize,
) -> (&[u32], &mut [u32]) {
    let (row1, row2) = if source_row < destination_row {
        (source_row, destination_row)
    } else {
        (destination_row, source_row)
    };
    let (out1, rest) = buf[row1 * width..].split_at_mut(width);
    let out2 = &mut rest[(row2 - row1 - 1) * width..(row2 - row1) * width];
    if source_row < destination_row {
        (out1, out2)
    } else {
        (out2, out1)
    }
}

struct Col<'a> {
    buf: &'a [u32],
    width: usize,
    col: usize,
}
struct ColMut<'a> {
    buf: &'a mut [u32],
    width: usize,
    col: usize,
}

impl<'a> ColMut<'a> {
    fn copy_from(&mut self, src: Col<'_>) {
        assert_eq!(self.width, src.width);
        assert_eq!(self.buf.as_ptr(), src.buf.as_ptr());
        assert_ne!(self.col, src.col);
        for i in (0..self.buf.len()).step_by(self.width) {
            self.buf[i + self.col] = src.buf[i + src.col];
        }
    }
}

fn select_cols(
    buf: &mut [u32],
    width: usize,
    source_col: usize,
    destination_col: usize,
) -> (Col, ColMut) {
    assert_ne!(source_col, destination_col);
    assert!(source_col < width);
    assert!(destination_col < width);

    // SAFETY: the two columns are nonoverlapping
    let buf_ref = unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.len()) };

    (
        Col {
            buf: buf_ref,
            width,
            col: source_col,
        },
        ColMut {
            buf,
            width,
            col: destination_col,
        },
    )
}

/// center_x and center_y are [0, 1)
fn zoom(
    buf: &mut [u32],
    height: usize,
    width: usize,
    center_x: f64,
    center_y: f64,
    multiplier: f64,
) {
    let center_row = (center_y * height as f64) as usize;

    let row_iter = if multiplier > 1.0 {
        (center_row + 1..height).rev().chain(0..center_row - 1)
    } else {
        (0..center_row - 1).rev().chain(center_row + 1..height)
    };

    for row in row_iter {
        let new_location = if row > center_row {
            ((row - center_row) as f64 * multiplier) as usize + center_row
        } else if let Some(nl) =
            center_row.checked_sub(((center_row - row) as f64 * multiplier) as usize)
        {
            nl
        } else {
            continue;
        };

        if new_location >= height {
            continue; // TODO: fix
        }

        if row > center_row && new_location <= center_row
            || row < center_row && new_location >= center_row
        {
            continue;
        }

        if new_location != row {
            let (src, dest) = select_rows(buf, width, row, new_location);
            dest.copy_from_slice(src);
        }
    }

    ////////////////////

    let center_col = (center_x * width as f64) as usize;

    let col_iter = if multiplier > 1.0 {
        (center_col + 1..width).rev().chain(0..center_col - 1)
    } else {
        (0..center_col - 1).rev().chain(center_col + 1..width)
    };

    for col in col_iter {
        let new_location = if col > center_col {
            ((col - center_col) as f64 * multiplier) as usize + center_col
        } else if let Some(nl) =
            center_col.checked_sub(((center_col - col) as f64 * multiplier) as usize)
        {
            nl
        } else {
            continue;
        };

        if new_location >= width {
            continue;
        }

        if col > center_col && new_location <= center_col
            || col < center_col && new_location >= center_col
        {
            continue;
        }

        if new_location != col {
            let (src, mut dest) = select_cols(buf, width, col, new_location);
            dest.copy_from(src);
        }
    }
}

#[derive(Copy, Clone)]
struct BufView(*mut u32);

unsafe impl Send for BufView {}
unsafe impl Sync for BufView {}

fn render<F: Float>(
    buf: &mut [u32],
    w: usize,
    h: usize,
    // the coordinate of the top-left corner (x, y)
    origin: (F, F),
    view: F,
) {
    let (x_min, y_min) = origin;
    let x_range = F::from_usize(w) * view;
    let y_range = F::from_usize(h) * view;

    let buf_view = BufView(buf.as_mut_ptr());

    let pbar = &ProgressBar::new(h as u64);
    (0..h).into_par_iter().for_each(move |r| {
        let buf_view = buf_view;

        for c in 0..w {
            let x = F::from_usize(c) / F::from_usize(w) * x_range + x_min;
            let y = F::from_usize(r) / F::from_usize(h) * y_range + y_min;
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

enum RenderPrecision {
    F16,
    F32,
    F64,
    // F128,
    F256,
}

impl Display for RenderPrecision {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderPrecision::F16 => writeln!(f, "f16"),
            RenderPrecision::F32 => writeln!(f, "f32"),
            RenderPrecision::F64 => writeln!(f, "f64"),
            // RenderPrecision::F128 => writeln!(f, "f128"),
            RenderPrecision::F256 => writeln!(f, "f256"),
        }
    }
}

fn main() {
    let w = 1000;
    let h = 600;

    let mut buffer: Vec<u32> = vec![0; w * h];

    let mut window = Window::new("Mandelbrot", w, h, WindowOptions::default()).unwrap();

    window.set_target_fps(60);

    let mut origin_x = -1.0;
    let mut origin_y = -1.0;
    // displayed range / pixel size (zooming in means reducing view)
    let mut view = 1.0 / 500.0;

    let mut cached = None;

    let mut first_time = true;
    let mut render_precision = RenderPrecision::F64;

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
        if window.is_key_down(Key::Space) {
            match render_precision {
                RenderPrecision::F16 => render::<f16>(&mut buffer, w, h, (origin_x as f16, origin_y as f16), view as f16),
                RenderPrecision::F32 => render::<f32>(&mut buffer, w, h, (origin_x as f32, origin_y as f32), view as f32),
                RenderPrecision::F64 => render::<f64>(&mut buffer, w, h, (origin_x, origin_y), view),
                // RenderPrecision::F128 => render::<f128>(&mut buffer, w, h, (origin_x as f128, origin_y as f128), view as f128),
                RenderPrecision::F256 => render::<f256>(&mut buffer, w, h, (origin_x.into(), origin_y.into()), view.into()),
            }
            cached = Some(cache_key);
        } else if window.is_key_pressed(Key::Escape, KeyRepeat::No) {
            render_precision = match render_precision {
                RenderPrecision::F16 => RenderPrecision::F32,
                RenderPrecision::F32 => RenderPrecision::F64,
                RenderPrecision::F64 => RenderPrecision::F256,
                // RenderPrecision::F64 => RenderPrecision::F128,
                // RenderPrecision::F128 => RenderPrecision::F16,
                RenderPrecision::F256 => RenderPrecision::F16,
            };
            println!("render precision: {render_precision}");
        } else {
            if cached != Some(cache_key) {
                if first_time {
                    render(&mut buffer, w, h, (origin_x, origin_y), view);
                    first_time = false;
                } else if let Some((center_x, center_y, multiplier)) = mouse01 {
                    zoom(&mut buffer, h, w, center_x, center_y, multiplier);
                }
                cached = Some(cache_key);
            }
        }

        window.update_with_buffer(&buffer, w, h).unwrap();
    }
}
