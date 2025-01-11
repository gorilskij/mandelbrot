use num::complex::{Complex};
use image::{ImageBuffer, Rgb};
use indicatif::ProgressBar;

const ITERATIONS: usize = 1000;

fn calculate(c: Complex<f64>) -> bool {
    let mut z: Complex<f64> = Complex::ZERO;
    for _ in 0..ITERATIONS {
        z = z * z + c;
        if z.is_infinite() || z.is_nan() { return false; }
    }
    !z.is_infinite() && !z.is_nan()
}

fn main() {
    let (w, h) = (32768, 32768);
    let mut buffer: ImageBuffer<Rgb<u8>, _> = ImageBuffer::new(w, h);

    let (x_min, x_max) = (-1.5, 0.5);
    let (y_min, y_max) = (-1.5, 1.5);

    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    let pbar = ProgressBar::new(h as u64);
    for r in 0..h {
        for c in 0..w {
            let x = r as f64 / w as f64 * x_range + x_min;
            let y = c as f64 / h as f64 * y_range + y_min;
            let ans = calculate(Complex::new(x, y));
            if ans {
                buffer.put_pixel(r, c, Rgb([255, 0, 0]));
            } else {
                buffer.put_pixel(r, c, Rgb([0, 0, 0]));
            }
        }
        pbar.inc(1);
    }

    println!("ELAPSED {:?}", pbar.elapsed());

    buffer.save("images/test.png").unwrap();
}
