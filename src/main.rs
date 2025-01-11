use std::f64::consts::E;
use num::complex::{Complex};
use image::{ImageBuffer, Rgb};
use indicatif::ProgressBar;
use rayon::prelude::*;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread;

const ITERATIONS: usize = 2000;

fn calculate(c: Complex<f64>) -> f64 {
    let mut z: Complex<f64> = Complex::ZERO;
    for i in 0..ITERATIONS {
        z = z * z + c;
        // if z.is_infinite() || z.is_nan() { return i as f64 / ITERATIONS as f64; }
        if z.norm() > 8.0 { return i as f64 / ITERATIONS as f64; }
    }
    if z.is_infinite() || z.is_nan() { return 1.0; }
    0.0
}

fn main() {
    let (w, h) = (10000, 10000);

    let (x_min, x_max) = (-1.5, 0.5);
    let (y_min, y_max) = (-1.0, 1.0);

    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    let (tx, rx) = channel::<(u32, Vec<f64>)>();

    let pbar = ProgressBar::new((h / 2 + 1) as u64);

    let buffer: ImageBuffer<Rgb<u8>, _> = ImageBuffer::new(w, h);
    let buffer = Arc::new(Mutex::new(buffer));
    let buffer_clone = Arc::clone(&buffer);
    let inserter = thread::spawn(move || {
        let mut buffer = buffer_clone.lock().unwrap();
        for _ in 0..h / 2 + 1 {
            let (r, row) = rx.recv().unwrap();
            for (c, val) in row.into_iter().enumerate() {
                let pixel = if val == 0.0 {
                    Rgb([0, 0, 0])
                } else {
                    // [0, 1]
                    let red = val;
                    let green = E.powf(-0.5 * ((val - 0.5) / 0.1).powi(2));
                    let blue = E.powf(-0.5 * ((val - 0.2) / 0.1).powi(2));

                    Rgb([
                        (red * 255.0) as u8,
                        (green * 255.0) as u8,
                        (blue * 255.0) as u8,
                    ])
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

    buffer.lock().unwrap().save("test.png").unwrap();
}
