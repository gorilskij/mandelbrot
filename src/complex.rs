use std::ops::{Add, Div, Mul, Sub};

pub trait Zero {
    const ZERO: Self;
}

impl Zero for f16 {
    const ZERO: Self = 0.0;
}

impl Zero for f32 {
    const ZERO: Self = 0.0;
}

impl Zero for f64 {
    const ZERO: Self = 0.0;
}

impl Zero for f128 {
    const ZERO: Self = 0.0;
}

pub trait Sqrt {
    fn sqrt(self) -> Self;
}

impl Sqrt for f16 {
    fn sqrt(self) -> Self {
        f16::sqrt(self)
    }
}

impl Sqrt for f32 {
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl Sqrt for f64 {
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

impl Sqrt for f128 {
    fn sqrt(self) -> Self {
        f128::sqrt(self)
    }
}

pub trait FromF64 {
    fn from_f64(n: f64) -> Self;
}

impl FromF64 for f16 {
    fn from_f64(n: f64) -> Self {
        n as f16
    }
}

impl FromF64 for f32 {
    fn from_f64(n: f64) -> Self {
        n as f32
    }
}

impl FromF64 for f64 {
    fn from_f64(n: f64) -> Self {
        n as f64
    }
}

impl FromF64 for f128 {
    fn from_f64(n: f64) -> Self {
        n as f128
    }
}

pub trait FromUsize {
    fn from_usize(n: usize) -> Self;
}

impl FromUsize for f16 {
    fn from_usize(n: usize) -> Self {
        n as f16
    }
}

impl FromUsize for f32 {
    fn from_usize(n: usize) -> Self {
        n as f32
    }
}

impl FromUsize for f64 {
    fn from_usize(n: usize) -> Self {
        n as f64
    }
}

impl FromUsize for f128 {
    fn from_usize(n: usize) -> Self {
        n as f128
    }
}

pub trait IntoF64 {
    fn into_f64(self) -> f64;
}

impl IntoF64 for f16 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}

impl IntoF64 for f32 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}

impl IntoF64 for f64 {
    fn into_f64(self) -> f64 {
        self
    }
}

impl IntoF64 for f128 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}

pub trait Float:
    Copy
    + Zero
    + Sqrt
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + PartialOrd
    + FromF64
    + FromUsize
    + IntoF64
    + Send
    + Sync
{
}

impl Float for f16 {}

impl Float for f32 {}

impl Float for f64 {}

impl Float for f128 {}

#[derive(Copy, Clone, Debug)]
pub struct Complex<F: Float> {
    re: F,
    im: F,
}

impl<F: Float> Zero for Complex<F> {
    const ZERO: Self = Self {
        re: F::ZERO,
        im: F::ZERO,
    };
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

impl<F: Float> Complex<F> {
    pub fn new(re: F, im: F) -> Self {
        Self { re, im }
    }

    pub fn norm(&self) -> F {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}
