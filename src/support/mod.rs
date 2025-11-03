mod euclid;

use dashu::float::{DBig, FBig};
pub use euclid::*;

pub trait ToFBig {
    fn to_fbig(self) -> FBig;
    fn to_fbig_with_precision(self, bits: usize) -> FBig;
}

impl ToFBig for f32 {
    fn to_fbig(self) -> FBig {
        self.to_string()
            .parse::<DBig>()
            .unwrap()
            .to_binary()
            .value()
    }

    fn to_fbig_with_precision(self, bits: usize) -> FBig {
        self.to_string()
            .parse::<DBig>()
            .unwrap()
            .to_binary()
            .value()
            .with_precision(bits)
            .value()
    }
}

impl ToFBig for f64 {
    fn to_fbig(self) -> FBig {
        self.to_string()
            .parse::<DBig>()
            .unwrap()
            .to_binary()
            .value()
    }

    fn to_fbig_with_precision(self, bits: usize) -> FBig {
        self.to_string()
            .parse::<DBig>()
            .unwrap()
            .to_binary()
            .value()
            .with_precision(bits)
            .value()
    }
}
