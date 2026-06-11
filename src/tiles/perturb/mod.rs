//! Perturbation backends.
//!
//! The orchestration in `tiles::render` enumerates the visible tiles, manages
//! the per-group reference lists, and drives the progressive passes. The
//! actual per-pixel perturbation evaluation is delegated to a `Perturbator`
//! backend. There are two sister implementations, each a unit struct:
//!   - [`cpu::Cpu`]  — f32 perturbation on the CPU (rayon).
//!   - [`gpu::Gpu`]  — wgpu compute.
//!
//! Reference *base points* are always high-precision FBig and computed on the
//! CPU (GPUs have no bignum); a backend only ever sees the projected `Pf`
//! orbits and f32 deltas.

pub mod cpu;
pub mod gpu;

use crate::rendering::{Orbit, Pf};
use crate::support::append_only::List as AOList;
use crate::tiles::store::Tile;
use num::Complex;
use waker_interrupter::MultiInterrupter;

/// A reference orbit (projected to `Pf`) together with its base point's delta
/// from the group anchor.
pub struct RefOrbit {
    pub delta_corr: Complex<Pf>,
    pub orbit: Orbit<Pf>,
}

/// A group's shared, growable reference list (lock-free append-only). All
/// tiles in a group read from it and promote new references into it.
pub type RefList = AOList<RefOrbit>;

/// Evaluates a tile's progressive pass against a group's reference list.
/// Implemented by the `cpu` and `gpu` sister modules.
pub trait Perturbator: Sync {
    /// Render one progressive pass of `tile` against `refs`, promoting new
    /// references for pixels that no existing reference can resolve.
    /// `anchor_px` is the group anchor's pixel offset within the group (deltas
    /// are measured from there). Returns `false` if interrupted (partial
    /// per-pixel progress is kept and skipped on retry).
    fn render_tile_pass(
        &self,
        tile: &Tile,
        refs: &RefList,
        anchor_px: (i64, i64),
        pass: u8,
        iterations: usize,
        int: &MultiInterrupter,
    ) -> bool;
}
