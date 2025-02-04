#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct MaybePixel(u32);

// incompatible with transparency
impl MaybePixel {
    const FLAG_BIT: u32 = 1 << 31;

    pub fn none() -> Self {
        MaybePixel(Self::FLAG_BIT)
    }

    pub fn get(self) -> Option<u32> {
        (self.0 & Self::FLAG_BIT == 0).then_some(self.0)
    }
}

impl From<u32> for MaybePixel {
    // creates a `Some` pixel
    fn from(value: u32) -> Self {
        Self(value & !Self::FLAG_BIT)
    }
}
