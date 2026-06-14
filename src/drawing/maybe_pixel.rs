#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct MaybePixel(u32);

// incompatible with transparency
// if the flag bit is set, this is a None, otherwise it's a Some(color)
impl MaybePixel {
    const FLAG_BIT: u32 = 1 << 31;

    /// raw representation of a `None` pixel
    pub const NONE_RAW: u32 = Self::FLAG_BIT;

    pub fn get(self) -> Option<u32> {
        (self.0 & Self::FLAG_BIT == 0).then_some(self.0)
    }

    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    pub fn to_raw(self) -> u32 {
        self.0
    }
}

impl From<u32> for MaybePixel {
    // creates a `Some` pixel
    fn from(value: u32) -> Self {
        Self(value & !Self::FLAG_BIT)
    }
}
