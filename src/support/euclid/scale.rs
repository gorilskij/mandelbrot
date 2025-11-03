use std::{
    fmt::Debug,
    marker::PhantomData,
    mem::transmute,
    ops::{Div, Mul},
};

#[repr(transparent)]
#[derive(Copy)]
pub struct Scale<T, FromUnit, ToUnit> {
    pub inner: T,
    _phantom: PhantomData<(FromUnit, ToUnit)>,
}

impl<T, FromUnit, ToUnit> Scale<T, FromUnit, ToUnit> {
    pub fn new(s: T) -> Self {
        Self {
            inner: s,
            _phantom: PhantomData,
        }
    }

    pub fn new_ref(s: &T) -> &Self {
        // SAFETY: repr(transparent)
        unsafe { transmute(s) }
    }

    pub fn cast<U>(&self, f: impl Fn(&T) -> U) -> Scale<U, FromUnit, ToUnit> {
        Scale::new(f(&self.inner))
    }
}

impl<T, FromUnit, ToUnit> Clone for Scale<T, FromUnit, ToUnit>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.inner.clone())
    }
}

impl<T, Unit1, Unit2, Unit3> Mul<Scale<T, Unit2, Unit3>> for Scale<T, Unit1, Unit2>
where
    T: Mul,
{
    type Output = Scale<T::Output, Unit1, Unit3>;

    fn mul(self, rhs: Scale<T, Unit2, Unit3>) -> Self::Output {
        Scale::new(self.inner * rhs.inner)
    }
}

impl<T, FromUnit, ToUnit> Div for Scale<T, FromUnit, ToUnit>
where
    T: Div,
{
    type Output = T::Output;

    fn div(self, rhs: Self) -> Self::Output {
        self.inner / rhs.inner
    }
}

pub type Length<T, Unit> = Scale<T, Unit, Unit>;

impl<T: Debug, FromUnit, ToUnit> Debug for Scale<T, FromUnit, ToUnit> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scale").field("inner", &self.inner).finish()
    }
}
