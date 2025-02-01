#![feature(test)]

extern crate test;

mod common;

use common::{ITERATIONS, run};
use num::Complex;
use test::Bencher;

fn calculate_component(c: Complex<f64>) -> Option<f64> {
    let mut z: Complex<f64> = Complex::ZERO;
    for i in 0..ITERATIONS {
        z = z * z + c;
        if z.re > 2.0 || z.im > 2.0 {
            return Some(i as f64 / ITERATIONS as f64);
        }
    }
    None
}

fn calculate_norm(c: Complex<f64>) -> Option<f64> {
    let mut z: Complex<f64> = Complex::ZERO;
    for i in 0..ITERATIONS {
        z = z * z + c;
        if z.norm() > 4.0 {
            return Some(i as f64 / ITERATIONS as f64);
        }
    }
    None
}

#[bench]
fn bench_calculate_component(b: &mut Bencher) {
    b.iter(|| test::black_box(run(calculate_component)))
}

#[bench]
fn bench_calculate_norm(b: &mut Bencher) {
    b.iter(|| test::black_box(run(calculate_norm)))
}
