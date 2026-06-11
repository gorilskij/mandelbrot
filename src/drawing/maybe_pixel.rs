#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct MaybePixel(u32);

// incompatible with transparency
// if the flag bit is set, this is a None, otherwise it's a Some(color)
impl MaybePixel {
    const FLAG_BIT: u32 = 1 << 31;

    pub fn none() -> Self {
        MaybePixel(Self::FLAG_BIT)
    }

    pub fn set_none(&mut self) {
        // self.0 = Self::FLAG_BIT
        // DEBUG
        self.0 = 0x00ffee_u32.into()
    }

    pub fn get(self) -> Option<u32> {
        (self.0 & Self::FLAG_BIT == 0).then_some(self.0)
    }
}

impl Default for MaybePixel {
    fn default() -> Self {
        Self::none()
    }
}

impl From<u32> for MaybePixel {
    // creates a `Some` pixel
    fn from(value: u32) -> Self {
        Self(value & !Self::FLAG_BIT)
    }
}
