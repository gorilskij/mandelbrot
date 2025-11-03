use num::Zero;

use super::*;
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub, SubAssign},
};

#[derive(Copy)]
pub struct Point<T, Unit> {
    pub x: T,
    pub y: T,
    _phantom: PhantomData<Unit>,
}

impl<T, Unit> Point<T, Unit> {
    pub fn new(x: T, y: T) -> Self {
        Self {
            x,
            y,
            _phantom: PhantomData,
        }
    }

    pub fn cast<U>(&self, f: impl Fn(&T) -> U) -> Point<U, Unit> {
        Point::new(f(&self.x), f(&self.y))
    }
}

impl<T: Clone + PartialOrd, Unit> Point<T, Unit> {
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        fn clamp<T: Clone + PartialOrd>(x: &T, min: &T, max: &T) -> T {
            assert!(min <= max);
            if x < min {
                min.clone()
            } else if x > max {
                max.clone()
            } else {
                x.clone()
            }
        }

        Self::new(
            clamp(&self.x, &min.x, &max.x),
            clamp(&self.y, &min.y, &max.y),
        )
    }
}

impl<T, Unit> Clone for Point<T, Unit>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: self.y.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, Unit> PartialEq for &'a Point<T, Unit>
where
    &'a T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        &self.x == &other.x && &self.y == &other.y
    }
}

impl<T: Zero, Unit> Zero for Point<T, Unit> {
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }
}

impl<T, Unit> Add for Point<T, Unit>
where
    T: Add,
{
    type Output = Point<<T as Add>::Output, Unit>;

    fn add(self, rhs: Self) -> Self::Output {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: 'a, Unit> Add for &'a Point<T, Unit>
where
    &'a T: Add,
{
    type Output = Point<<&'a T as Add>::Output, Unit>;

    fn add(self, rhs: Self) -> Self::Output {
        Point {
            x: &self.x + &rhs.x,
            y: &self.y + &rhs.y,
            _phantom: PhantomData,
        }
    }
}

impl<T, Unit> Sub for Point<T, Unit>
where
    T: Sub,
{
    type Output = Point<<T as Sub>::Output, Unit>;

    fn sub(self, rhs: Self) -> Self::Output {
        Point {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: 'a, Unit> Sub for &'a Point<T, Unit>
where
    &'a T: Sub,
{
    type Output = Point<<&'a T as Sub>::Output, Unit>;

    fn sub(self, rhs: Self) -> Self::Output {
        Point {
            x: &self.x - &rhs.x,
            y: &self.y - &rhs.y,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: 'a, Unit> SubAssign<&'a Point<T, Unit>> for Point<T, Unit>
where
    T: Clone,
    T: Sub<&'a T, Output = T>,
{
    fn sub_assign(&mut self, rhs: &'a Self) {
        let this = self.clone();
        self.x = this.x - &rhs.x;
        self.y = this.y - &rhs.y;
    }
}

impl<T, FromUnit, ToUnit> Mul<Scale<T, FromUnit, ToUnit>> for Point<T, FromUnit>
where
    T: Copy + Mul,
{
    type Output = Point<<T as Mul>::Output, ToUnit>;

    fn mul(self, rhs: Scale<T, FromUnit, ToUnit>) -> Self::Output {
        Point {
            x: self.x * rhs.inner,
            y: self.y * rhs.inner,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: 'a, FromUnit, ToUnit> Mul<&'a Scale<T, FromUnit, ToUnit>> for &'a Point<T, FromUnit>
where
    &'a T: Mul,
{
    type Output = Point<<&'a T as Mul>::Output, ToUnit>;

    fn mul(self, rhs: &'a Scale<T, FromUnit, ToUnit>) -> Self::Output {
        Point {
            x: &self.x * &rhs.inner,
            y: &self.y * &rhs.inner,
            _phantom: PhantomData,
        }
    }
}

impl<T, FromUnit, ToUnit> Div<Scale<T, FromUnit, ToUnit>> for Point<T, ToUnit>
where
    T: Copy + Div,
{
    type Output = Point<<T as Div>::Output, FromUnit>;

    fn div(self, rhs: Scale<T, FromUnit, ToUnit>) -> Self::Output {
        Point {
            x: self.x / rhs.inner,
            y: self.y / rhs.inner,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: 'a, FromUnit, ToUnit> Div<&'a Scale<T, FromUnit, ToUnit>> for &'a Point<T, ToUnit>
where
    &'a T: Div,
{
    type Output = Point<<&'a T as Div>::Output, FromUnit>;

    fn div(self, rhs: &'a Scale<T, FromUnit, ToUnit>) -> Self::Output {
        Point {
            x: &self.x / &rhs.inner,
            y: &self.y / &rhs.inner,
            _phantom: PhantomData,
        }
    }
}

impl<T, Unit> Mul<T> for Point<T, Unit>
where
    T: Copy + Mul,
{
    type Output = Point<<T as Mul>::Output, Unit>;

    fn mul(self, rhs: T) -> Self::Output {
        self * Length::new(rhs)
    }
}

impl<'a, T: 'a, Unit> Mul<&'a T> for &'a Point<T, Unit>
where
    &'a T: Mul,
{
    type Output = Point<<&'a T as Mul>::Output, Unit>;

    fn mul(self, rhs: &'a T) -> Self::Output {
        self * Length::new_ref(rhs)
    }
}

impl<T, Unit> Div<T> for Point<T, Unit>
where
    T: Copy + Div,
{
    type Output = Point<<T as Div>::Output, Unit>;

    fn div(self, rhs: T) -> Self::Output {
        self / Length::new(rhs)
    }
}

impl<'a, T: 'a, Unit> Div<&'a T> for &'a Point<T, Unit>
where
    &'a T: Div,
{
    type Output = Point<<&'a T as Div>::Output, Unit>;

    fn div(self, rhs: &'a T) -> Self::Output {
        self / Length::new_ref(rhs)
    }
}

impl<T: Debug, Unit> Debug for Point<T, Unit> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Point")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}
