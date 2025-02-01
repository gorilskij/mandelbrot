use num::Complex;

pub const ITERATIONS: usize = 10_000;

const LEN: usize = 10;

pub fn run(calculate: fn(Complex<f64>) -> Option<f64>) -> Vec<f64> {
    let mut buf = vec![0.0; LEN * LEN];
    (0..LEN).for_each(|re| {
        (0..LEN).for_each(|im| {
            let val = calculate(Complex {
                re: (re as f64) / const { LEN as f64 } - 0.5,
                im: (im as f64) / const { LEN as f64 } - 0.5,
            });
            buf[re * LEN + im] = val.unwrap_or(0.0);
        })
    });
    buf
}
