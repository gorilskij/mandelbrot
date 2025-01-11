use std::f64::consts::E;
use num::complex::Complex;
use image::{ImageBuffer, Rgb, imageops::sample_bilinear};
use indicatif::ProgressBar;
use rayon::prelude::*;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread;
use hsl::HSL;

const ITERATIONS: usize = 2000;
const OVERSAMPLE: u32 = 1;

fn calculate(c: Complex<f64>) -> (f64, f64) {
    let mut to_mid = None;

    let mut z: Complex<f64> = Complex::ZERO;
    for i in 0..ITERATIONS {
        z = z * z + c;
        if to_mid.is_none() && z.norm() > 4.0 {
            to_mid = Some(i as f64 / ITERATIONS as f64);
        }
        if z.is_infinite() || z.is_nan() {
            let to_inf = i as f64 / ITERATIONS as f64;
            let to_mid = to_mid.unwrap_or(to_inf);
            return (to_mid, to_inf);
        }
    }
    (0.0, 0.0)
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

fn main() {
    let (w, h) = (12000, 12000);

    let (x_min, x_max) = (-2.5, 1.0);
    let (y_min, y_max) = (-1.75, 1.75);

    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    let (tx, rx) = channel::<(u32, Vec<(f64, f64)>)>();

    let pbar = ProgressBar::new((h / 2 + 1) as u64);

    let buffer: ImageBuffer<Rgb<u8>, _> = ImageBuffer::new(w, h);
    let buffer = Arc::new(Mutex::new(buffer));
    let buffer_clone = Arc::clone(&buffer);
    let inserter = thread::spawn(move || {
        let mut buffer = buffer_clone.lock().unwrap();
        for _ in 0..h / 2 + 1 {
            let (r, row) = rx.recv().unwrap();
            for (c, val) in row.into_iter().enumerate() {
                let pixel = if val == (0.0, 0.0) {
                    Rgb([0, 0, 0])
                } else {
                    let h = apply_sharpness(val.0, 30.0) / 6.0;
                    let l = apply_sharpness(val.1, 20.0) * 0.6;

                    let hsl = HSL {
                        h: h * 360.0,
                        s: 1.0,
                        l,
                    };

                    let (r, g, b) = hsl.to_rgb();

                    Rgb([r, g, b])
                };

                buffer.put_pixel(c as u32, r, pixel);
                buffer.put_pixel(c as u32, h - r - 1, pixel);
            }
        }
    });

    (0..h / 2 + 1).into_par_iter().for_each(|r| {
        let mut row = Vec::with_capacity(w as usize);
        for c in 0..w {
            let x = c as f64 / w as f64 * x_range + x_min;
            let y = r as f64 / h as f64 * y_range + y_min;
            let val = calculate(Complex::new(x, y));
            row.push(val);
        }
        tx.send((r, row)).unwrap();
        pbar.inc(1);
    });

    inserter.join().unwrap();

    pbar.finish();
    println!("ELAPSED {:?}", pbar.elapsed());

    let buffer = buffer.lock().unwrap();

    let mut new_buffer = ImageBuffer::new(w * OVERSAMPLE, h * OVERSAMPLE);
    for r in 0..h * OVERSAMPLE {
        for c in 0..w * OVERSAMPLE {
            let y = r as f32 / (h * OVERSAMPLE) as f32;
            let x = c as f32 / (w * OVERSAMPLE) as f32;
            let px = sample_bilinear(&*buffer, x, y).unwrap();
            new_buffer.put_pixel(c, r, px);
        }
    }

    new_buffer.save("test0.png").unwrap();
}
