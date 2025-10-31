use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Neg, Sub},
};

// represents val * 2^exp
#[derive(Clone, Copy)]
pub struct ExtendedFloat {
    val: f32,
    exp: i32,
}

impl ExtendedFloat {
    pub fn val(&self) -> f32 {
        self.val
    }

    pub fn exp(&self) -> i32 {
        self.exp
    }

    // make sure value is around 1
    fn normalize(&mut self) {
        if self.val().abs() <= f32::EPSILON {
            self.val = 0.0;
            self.exp = 1;
            return;
        }

        let power = self.val.log2() as i32;
        self.val /= 2_f32.powi(power as i32);
        self.exp += power;
    }

    pub fn new(val: f32) -> Self {
        let mut this = Self { val, exp: 0 };
        this.normalize();
        this
    }

    pub fn abs(&self) -> Self {
        Self {
            val: self.val.abs(),
            exp: self.exp,
        }
    }

    pub fn into_f64(&self) -> f64 {
        self.val as f64 * 2_f64.powi(self.exp)
    }

    pub fn sqrt(&self) -> Self {
        let mut out = Self {
            val: self.val.sqrt(),
            exp: self.exp,
        };
        out.normalize();
        out
    }
}

impl Neg for ExtendedFloat {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            val: -self.val,
            exp: self.exp,
        }
    }
}

impl Add for ExtendedFloat {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // bump both numbers to the higher exponent
        let mut out = if self.exp <= rhs.exp {
            Self {
                val: self.val * 2_f32.powi(self.exp - rhs.exp) + rhs.val,
                exp: rhs.exp,
            }
        } else {
            Self {
                val: self.val + rhs.val * 2_f32.powi(rhs.exp - self.exp),
                exp: self.exp,
            }
        };
        out.normalize();
        out
    }
}

impl Sub for ExtendedFloat {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl Mul for ExtendedFloat {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = Self {
            val: self.val * rhs.val,
            exp: self.exp + rhs.exp,
        };
        out.normalize();
        out
    }
}

impl Div for ExtendedFloat {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut out = Self {
            val: self.val / rhs.val,
            exp: self.exp - rhs.exp,
        };
        out.normalize();
        out
    }
}

impl PartialEq for ExtendedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val && self.exp == other.exp
    }
}

impl PartialOrd for ExtendedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.exp.partial_cmp(&other.exp) {
            Some(Ordering::Equal) => {}
            ord => return ord,
        }
        self.val.partial_cmp(&other.val)
    }
}

// #[derive(Clone, Copy)]
// pub struct ExtendedComplex {
//     re: ExtendedFloat,
//     im: ExtendedFloat,
// }

// impl ExtendedComplex {
//     pub fn new(re: f32, im: f32) -> Self {
//         Self {
//             re: ExtendedFloat::new(re),
//             im: ExtendedFloat::new(im),
//         }
//     }
// }

// impl Add for ExtendedComplex {
//     type Output = Self;

//     fn add(self, rhs: Self) -> Self::Output {
//         Self {
//             re: self.re + rhs.re,
//             im: self.im + rhs.im,
//         }
//     }
// }

// impl Mul for ExtendedComplex {
//     type Output = Self;

//     fn mul(self, rhs: Self) -> Self::Output {
//         Self {
//             re: self.re * rhs.re - self.im * rhs.im,
//             im: self.im * rhs.re + self.re * rhs.im,
//         }
//     }
// }
