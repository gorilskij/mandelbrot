use f256::f256;
use num_bigfloat::BigFloat;
use std::ops::{Add, Div, Mul, Sub};

use crate::extended_float::ExtendedFloat;

pub trait Zero {
    fn zero() -> Self;
}

impl Zero for f16 {
    fn zero() -> Self {
        0.0
    }
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

impl Zero for f128 {
    fn zero() -> Self {
        0.0
    }
}

impl Zero for f256 {
    fn zero() -> Self {
        f256::ZERO
    }
}

impl Zero for BigFloat {
    fn zero() -> Self {
        BigFloat::from_f64(0.0)
    }
}

impl Zero for ExtendedFloat {
    fn zero() -> Self {
        ExtendedFloat::new(0.0)
    }
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

impl Sqrt for f256 {
    fn sqrt(self) -> Self {
        f256::sqrt(self)
    }
}

impl Sqrt for BigFloat {
    fn sqrt(self) -> Self {
        BigFloat::sqrt(&self)
    }
}

impl Sqrt for ExtendedFloat {
    fn sqrt(self) -> Self {
        ExtendedFloat::sqrt(&self)
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
        n
    }
}

impl FromF64 for f128 {
    fn from_f64(n: f64) -> Self {
        n as f128
    }
}

impl FromF64 for f256 {
    fn from_f64(n: f64) -> Self {
        f256::from(n)
    }
}

impl FromF64 for BigFloat {
    fn from_f64(n: f64) -> Self {
        Self::from(n)
    }
}

impl FromF64 for ExtendedFloat {
    // TODO: redo
    fn from_f64(n: f64) -> Self {
        ExtendedFloat::new(n as f32)
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

impl FromUsize for f256 {
    fn from_usize(n: usize) -> Self {
        f256::from(n as f64)
    }
}

impl FromUsize for BigFloat {
    fn from_usize(n: usize) -> Self {
        Self::from(n as f64)
    }
}

impl FromUsize for ExtendedFloat {
    // TODO: redo
    fn from_usize(n: usize) -> Self {
        Self::new(n as f32)
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

impl IntoF64 for f256 {
    fn into_f64(self) -> f64 {
        if self.is_nan() {
            return f64::NAN;
        }
        if self.is_infinite() {
            return if self.is_sign_positive() {
                f64::INFINITY
            } else {
                -f64::INFINITY
            };
        }

        let (hi, lo) = self.to_bits();
        let sign = hi >> 127;
        let exp = (hi >> 108) & (!(1 << 20));
        if exp > 0b0111_1111_1111
        /* 11 bits */
        {
            return if sign == 0 {
                f64::INFINITY
            } else {
                -f64::INFINITY
            };
        }
        let mant = (0x0fff_ffff_ffff_ff00_0000_0000_0000 /* first 52 bits */ & hi) >> 56;

        let float = ((sign as u64) << 63) | ((exp as u64) << 52) | mant as u64;

        f64::from_bits(float)
    }
}

impl IntoF64 for BigFloat {
    fn into_f64(self) -> f64 {
        self.to_f64()
    }
}

impl IntoF64 for ExtendedFloat {
    fn into_f64(self) -> f64 {
        ExtendedFloat::into_f64(&self)
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
    fn abs(self) -> Self;
}

impl Float for f16 {
    fn abs(self) -> Self {
        f16::abs(self)
    }
}

impl Float for f32 {
    fn abs(self) -> Self {
        f32::abs(self)
    }
}

impl Float for f64 {
    fn abs(self) -> Self {
        f64::abs(self)
    }
}

impl Float for f128 {
    fn abs(self) -> Self {
        f128::abs(self)
    }
}

impl Float for f256 {
    fn abs(self) -> Self {
        f256::abs(&self)
    }
}

impl Float for BigFloat {
    fn abs(self) -> Self {
        BigFloat::abs(&self)
    }
}

impl Float for ExtendedFloat {
    fn abs(self) -> Self {
        ExtendedFloat::abs(&self)
    }
}

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
