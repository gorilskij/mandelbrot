#![feature(f16)]

mod complex;

use complex::{Complex, Float, Zero};
use hsl::HSL;
use indicatif::ProgressBar;
use minifb::{Key, KeyRepeat, MouseMode, Window, WindowOptions};
use num_bigfloat::BigFloat;
use rayon::prelude::*;

use crate::complex::IntoF64;

const ITERATIONS: usize = 1000;

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
    !(v.re().abs() <= 1000.0 && v.im().abs() <= 1000.0)
}

// fn is_bad_value(v: Complex<BigFloat>) -> bool {
//     // implicitly also checks that neither is NaN
//     let k = BigFloat::from(1000.0);
//     !(v.re().abs() <= k && v.im().abs() <= k)
// }

// fn calculate_delta_orbit(
//     ref_orbit: &[Complex<BigFloat>],
//     delta: Complex<BigFloat>,
// ) -> Result<(Vec<Complex<BigFloat>>, Vec<Complex<BigFloat>>), ()> {
//     // ref_orbit is [x_0, x_1, ...]
//     // delta is delta_0
//     // delta_{n+1} = 2 x_n delta_n + delta_n^2 + delta_0

//     let mut deltas = vec![];
//     let mut out = vec![];

//     let delta_0 = delta;
//     let mut delta = delta;
//     for i in 0..ref_orbit.len() {
//         if is_bad_value(ref_orbit[i]) {
//             return Err(());
//         }

//         let cur_estimate = ref_orbit[i] + delta;

//         deltas.push(delta);
//         out.push(cur_estimate);
//         let ee = ref_orbit[i] * BigFloat::from(2.0) * delta + delta * delta + delta_0;
//         delta = ee;
//     }
//     Ok((deltas, out))
// }

fn check_orbit<F: Float>(orbit: &[Complex<F>]) -> Option<usize> {
    for (i, x) in orbit.iter().enumerate() {
        if x.norm() > F::from_f64(4.0) {
            return Some(i);
        }
    }
    None
}

fn check_divergence_delta(
    ref_orbit: &[Complex<BigFloat>],
    ref_orbit_f64: &[Complex<f64>],
    delta: Complex<f64>,
) -> Result<Option<usize>, ()> {
    // ref_orbit is [x_0, x_1, ...]
    // delta is delta_0
    // delta_{n+1} = 2 x_n delta_n + delta_n^2 + delta_0

    let delta_0 = delta;
    let mut delta = delta;
    for i in 0..ref_orbit.len() {
        if is_bad_value(ref_orbit_f64[i]) {
            return Err(());
        }

        // let val =
        // ref_orbit[i] + Complex::new(BigFloat::from(delta.re()), BigFloat::from(delta.im()));
        let x = ref_orbit_f64[i] + delta;
        if (x.re() > 2.0 || x.im() > 2.0) && x.norm() > 4.0 {
            return Ok(Some(i));
        }

        delta = ref_orbit_f64[i] * 2.0 * delta + delta * delta + delta_0;
    }
    Ok(None)
}

fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

fn render_pixel(val: Option<usize>) -> u32 {
    if let Some(val) = val {
        // [0, 1)
        let f = 1.0 - 1.0 / (val as f64 / 100.0 + 1.0);

        let hsl = HSL {
            // h: val.get() as f64 % 360.0,
            // s: 0.7,
            // // l: 0.5,
            // l: (val.get() as f64 / 10.0).sin() * 0.1 + 0.5,
            h: (val as f64 / 10.0).sin() * 180.0,
            s: 0.7,
            // l: 0.5,
            l: (val as f64 / (10.0 * std::f64::consts::E)).sin() * 0.4 + 0.5,
        };

        let (r, g, b) = hsl.to_rgb();
        rgb_to_u32(r, g, b)
    } else {
        0
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
) -> (Col<'_>, ColMut<'_>) {
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

fn render(
    buf: &mut [u32],
    w: usize,
    h: usize,
    // the coordinate of the top-left corner (x, y)
    origin: (BigFloat, BigFloat),
    view: f64,
) {
    let (x_min, y_min) = origin;
    let x_range = w as f64 * view;
    let y_range = h as f64 * view;

    let buf_view = BufView(buf.as_mut_ptr());

    let mut ref_orbit = calculate_orbit(
        Complex::new(BigFloat::from(x_min), BigFloat::from(y_min)),
        ITERATIONS,
    );
    let mut ref_orbit_f64 = Vec::from_iter(
        ref_orbit
            .iter()
            .map(|x| Complex::new(x.re().into_f64(), x.im().into_f64())),
    );

    let mut delta_corr_x = 0.0;
    let mut delta_corr_y = 0.0;

    let pbar = &ProgressBar::new(h as u64);
    // (0..h).into_par_iter().for_each(move |r| {
    (0..h).into_iter().for_each(move |r| {
        let buf_view = buf_view;

        for c in 0..w {
            let delta_x = c as f64 / w as f64 * x_range;
            let delta_y = r as f64 / h as f64 * y_range;

            let val = if let Ok(val) = check_divergence_delta(
                &ref_orbit,
                &ref_orbit_f64,
                Complex::new(delta_x - delta_corr_x, delta_y - delta_corr_y),
            ) {
                val
            } else {
                println!("recalculating reference");
                ref_orbit = calculate_orbit(
                    Complex::new(
                        x_min + BigFloat::from(delta_x),
                        y_min + BigFloat::from(delta_y),
                    ),
                    ITERATIONS,
                );
                ref_orbit_f64 = ref_orbit
                    .iter()
                    .map(|x| Complex::new(x.re().into_f64(), x.im().into_f64()))
                    .collect();
                delta_corr_x = delta_x;
                delta_corr_y = delta_y;
                check_orbit(&ref_orbit)
            };

            // let val = check_orbit(&calculate_orbit(
            //     Complex::new(
            //         x_min + BigFloat::from(delta_x),
            //         y_min + BigFloat::from(delta_y),
            //     ),
            //     ITERATIONS,
            // ));

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

fn pt(c: &Complex<BigFloat>) -> String {
    format!("({}, {})", c.re().into_f64(), c.im().into_f64())
}

// fn main_() {
//     let X = Complex::new(
//         BigFloat::from(-0.6337777492436207),
//         BigFloat::from(-0.9334837812869803),
//     );
//     let Y = Complex::new(
//         BigFloat::from(-0.2016548049962961),
//         BigFloat::from(-0.6742100147385854),
//     );

//     let D = Complex::new(Y.re() - X.re(), Y.im() - X.im());

//     println!("D: {:?}\n\n", D);

//     let o1 = calculate_orbit(X, 100);
//     let o2 = calculate_orbit(Y, 100);
//     if let Ok((deltas, o2_x)) = calculate_delta_orbit(&o1, D) {
//         assert_eq!(o1.len(), o2_x.len());

//         for (((a, b), d), x) in o2.iter().zip(&o2_x).zip(&deltas).zip(&o1) {
//             println!("X_n  {}", pt(x));
//             println!("Y_n  {}", pt(a));
//             println!("Y_n' {}", pt(b));
//             println!("D_n  {}", pt(d));
//             println!();
//         }
//     } else {
//         println!("error")
//     }
// }

fn main() {
    let w = 1000;
    let h = 600;

    let mut buffer: Vec<u32> = vec![0; w * h];

    let mut window = Window::new("Mandelbrot", w, h, WindowOptions::default()).unwrap();

    window.set_target_fps(60);

    let mut origin_x = BigFloat::from(-1.0);
    let mut origin_y = BigFloat::from(-1.0);
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

                let pos_x = BigFloat::from(mouse_x as f64 * view) + origin_x;
                let pos_y = BigFloat::from(mouse_y as f64 * view) + origin_y;

                origin_x = pos_x + (origin_x - pos_x) / BigFloat::from(multiplier);
                origin_y = pos_y + (origin_y - pos_y) / BigFloat::from(multiplier);
            }
        }

        let cache_key = (origin_x, origin_y, view);
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            render(
                &mut buffer,
                w,
                h,
                (origin_x.into(), origin_y.into()),
                view.into(),
            );
            cached = Some(cache_key);
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
