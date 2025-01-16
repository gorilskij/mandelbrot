use hsl::HSL;
use indicatif::ProgressBar;
use minifb::{Key, MouseMode, Window, WindowOptions};
use num::complex::Complex;
use rayon::prelude::*;

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

// Returns (&source, &mut destination)
fn select_rows(buf: &mut [u32], width: usize, source_row: usize, destination_row: usize) -> (&[u32], &mut [u32]) {
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

fn select_cols(buf: &mut [u32], width: usize, source_col: usize, destination_col: usize) -> (Col, ColMut) {
    assert_ne!(source_col, destination_col);
    assert!(source_col < width);
    assert!(destination_col < width);

    // SAFETY: the two columns are nonoverlapping
    let buf_ref = unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.len()) };

    (Col { buf: buf_ref, width, col: source_col }, ColMut { buf, width, col: destination_col })
}

/// center_x and center_y are [0, 1)
fn zoom(buf: &mut [u32], height: usize, width: usize, center_x: f64, center_y: f64, multiplier: f64) {
    if multiplier > 1.0 {
        // zooming in

        let center_row = (center_y * height as f64) as usize;

        for row in (center_row + 1..height).rev().chain(0..center_row - 1) {
            let new_location = if row > center_row {
                ((row - center_row) as f64 * multiplier) as usize + center_row
            } else if let Some(nl) = center_row.checked_sub(((center_row - row) as f64 * multiplier) as usize) {
                nl
            } else {
                continue;
            };

            if new_location >= height {
                continue; // TODO: fix
            }

            if new_location != row {
                let (src, dest) = select_rows(buf, width, row, new_location);
                dest.copy_from_slice(src);
            }
        }

        let center_col = (center_x * width as f64) as usize;

        for col in (center_col + 1..width).rev().chain(0..center_col - 1) {
            let new_location = if col > center_col {
                ((col - center_col) as f64 * multiplier) as usize + center_col
            } else if let Some(nl) = center_col.checked_sub(((center_col - col) as f64 * multiplier) as usize) {
                nl
            } else {
                continue;
            };

            if new_location >= width {
                continue;
            }

            if new_location != col {
                let (src, mut dest) = select_cols(buf, width, col, new_location);
                dest.copy_from(src);
            }
        }
    } else if multiplier < 1.0 {
        todo!()
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
                buf_view.0.offset((r * w + c) as isize).write(render_pixel(val));
            }
        }
        pbar.inc(1);
    });
    pbar.finish();
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

    while window.is_open() {
        let mut mouse01 = None;

        if let Some((mouse_x, mouse_y)) = window.get_mouse_pos(MouseMode::Discard) {
            if let Some((_, scroll_y)) = window.get_scroll_wheel() {
                let multiplier = 1.0 + (scroll_y as f64 / 100.0).clamp(-0.2, 0.2);

                mouse01 = Some((mouse_x as f64 / w as f64, mouse_y as f64 / h as f64, multiplier));

                view /= multiplier;

                let pos_x = mouse_x as f64 * view + origin_x;
                let pos_y = mouse_y as f64 * view + origin_y;

                origin_x = pos_x + (origin_x - pos_x) / multiplier;
                origin_y = pos_y + (origin_y - pos_y) / multiplier;
            }
        }

        let cache_key = (origin_x, origin_y, view);
        if cached != Some(cache_key) {
            if first_time {
                render(&mut buffer, w, h, (origin_x, origin_y), view);
                first_time = false;
            } else if let Some((center_x, center_y, multiplier)) = mouse01 {
                zoom(&mut buffer, h, w, center_x, center_y, multiplier);
            }
            cached = Some(cache_key);
        }

        window.update_with_buffer(&buffer, w, h).unwrap();
    }
}
