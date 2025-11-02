use num_traits::{Float, Zero};
use std::ops::{Add, Mul};

#[derive(Copy, Clone, Debug)]
pub struct Complex<F: Float> {
    re: F,
    im: F,
}

impl<F: Float> Zero for Complex<F> {
    fn zero() -> Self {
        Self {
            re: F::zero(),
            im: F::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.re().is_zero() && self.im().is_zero()
    }
}

impl<F: Float> Add for Complex<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<F: Float> Mul for Complex<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<F: Float> Mul<F> for Complex<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl Mul<Complex<f64>> for f64 {
    type Output = Complex<f64>;

    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        Complex {
            re: self * rhs.re,
            im: self * rhs.im,
        }
    }
}

impl<F: Float> Complex<F> {
    pub fn new(re: F, im: F) -> Self {
        Self { re, im }
    }

    pub fn re(&self) -> F {
        self.re
    }

    pub fn im(&self) -> F {
        self.im
    }

    pub fn norm(&self) -> F {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}
