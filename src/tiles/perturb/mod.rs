//! CPU perturbation backend (the only backend used by the tile pipeline).
//!
//! The GPU fullscreen path lives in `crate::gpu_compute` and bypasses the
//! tile system entirely — so `RefOrbit`, `RefList`, and `Perturbator` are
//! CPU-only concerns.

pub mod cpu;

use crate::rendering::{Orbit, Pf};
use crate::support::append_only::List as AOList;
use crate::tiles::store::Tile;
use num::Complex;
use waker_interrupter::MultiInterrupter;

/// A reference orbit (projected to `Pf`) together with its base point's delta
/// from the group anchor.
pub struct RefOrbit {
    pub delta_corr: Complex<Pf>,
    pub orbit:      Orbit<Pf>,
}

/// A group's shared, growable reference list (lock-free append-only). All
/// tiles in a group read from it and promote new references into it.
pub type RefList = AOList<RefOrbit>;

/// Evaluates a tile's progressive pass against a group's reference list.
pub trait Perturbator: Sync {
    fn render_tile_pass(
        &self,
        tile:       &Tile,
        refs:       &RefList,
        anchor_px:  (i64, i64),
        pass:       u8,
        iterations: usize,
        int:        &MultiInterrupter,
    ) -> bool;
}
