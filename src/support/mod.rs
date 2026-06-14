pub mod append_only;
mod euclid;

use dashu::float::FBig;
use dashu::integer::IBig;
pub use euclid::*;

/// Convert to an `FBig` at a given precision, exactly. Floats go through
/// `FBig::try_from` (mantissa/exponent decode), never a lossy decimal
/// round-trip. For an exact conversion at native precision use
/// `FBig::try_from(x)` directly.
pub trait ToFBig {
    fn to_fbig_with_precision(self, bits: usize) -> FBig;
}

impl ToFBig for f32 {
    fn to_fbig_with_precision(self, bits: usize) -> FBig {
        FBig::try_from(self).unwrap().with_precision(bits).value()
    }
}

impl ToFBig for f64 {
    fn to_fbig_with_precision(self, bits: usize) -> FBig {
        FBig::try_from(self).unwrap().with_precision(bits).value()
    }
}

impl ToFBig for usize {
    fn to_fbig_with_precision(self, bits: usize) -> FBig {
        FBig::from_parts(IBig::from(self), 0)
            .with_precision(bits)
            .value()
    }
}

// unused
// pub fn read_line_stdin() -> Result<Option<String>, ()> {
//     let mut line = String::new();
//     std::io::stdin().read_line(&mut line).map_err(|_| ())?;
//     let trimmed = line.trim();
//     if trimmed.is_empty() {
//         Ok(None)
//     } else {
//         Ok(Some(trimmed.to_string()))
//     }
// }
