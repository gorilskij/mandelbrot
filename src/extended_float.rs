use std::{
    cmp::Ordering,
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Sub},
};

// represents val * 2^exp
#[derive(Clone, Copy)]
pub struct ExtendedFloat {
    val: f16,
    exp: i32,
}

impl Debug for ExtendedFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}*2^{}", self.val, self.exp)
    }
}

impl ExtendedFloat {
    pub fn val(&self) -> f16 {
        self.val
    }

    pub fn exp(&self) -> i32 {
        self.exp
    }

    // make sure value is around 1
    fn normalize(&mut self) {
        if self.val().abs() <= f16::EPSILON {
            self.val = 0.0;
            self.exp = 1;
            return;
        }

        let power = self.val.log2() as i32;
        self.val /= 2_f16.powi(power as i32);
        self.exp += power;
    }

    pub fn new(val: f16) -> Self {
        let mut this = Self { val, exp: 0 };
        this.normalize();
        this
    }

    pub fn from_f64(val: f64) -> Self {
        if val.abs() <= f64::EPSILON {
            return Self { val: 0.0, exp: 1 };
        }

        let power = val.log2() as i32;
        Self {
            val: (val / 2_f64.powi(power as i32)) as f16,
            exp: power,
        }
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
                val: self.val * 2_f16.powi(self.exp - rhs.exp) + rhs.val,
                exp: rhs.exp,
            }
        } else {
            Self {
                val: self.val + rhs.val * 2_f16.powi(rhs.exp - self.exp),
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
