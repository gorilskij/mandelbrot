#![feature(test)]

extern crate test;

mod common;

use common::{ITERATIONS, run};
use num::Complex;
use test::Bencher;

fn calculate_loop(c: Complex<f64>) -> Option<f64> {
    let mut z: Complex<f64> = Complex::ZERO;
    for i in 0..ITERATIONS {
        z = z * z + c;
        if z.re > 2.0 || z.im > 2.0 {
            return Some(i as f64 / ITERATIONS as f64);
        }
    }
    None
}

fn calculate_iter(c: Complex<f64>) -> Option<f64> {
    let mut z: Complex<f64> = Complex::ZERO;

    let out = (0..ITERATIONS).try_for_each(move |i| {
        z = z * z + c;
        if z.re > 2.0 || z.im > 2.0 {
            Err(i as f64 / ITERATIONS as f64)
        } else {
            Ok(())
        }
    });
    out.err()
}

#[bench]
fn bench_calculate_loop(b: &mut Bencher) {
    b.iter(|| test::black_box(run(calculate_loop)))
}

#[bench]
fn bench_calculate_iter(b: &mut Bencher) {
    b.iter(|| test::black_box(run(calculate_iter)))
}
