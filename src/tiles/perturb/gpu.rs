//! GPU perturbation backend (wgpu compute) — step 2, not yet implemented.
//!
//! Planned shape: the reference orbits (`Pf` arrays + `delta_corr` + is_full)
//! are uploaded to a storage buffer, a compute shader runs the delta
//! iteration per pixel, and glitched pixels are returned as a mask so the CPU
//! can promote new references and re-dispatch (glitch-correction passes,
//! since promotion needs FBig and can't happen inside the kernel).

use super::{Perturbator, RefList};
use crate::tiles::store::Tile;
use waker_interrupter::MultiInterrupter;

/// wgpu perturbation backend.
pub struct Gpu;

impl Perturbator for Gpu {
    fn render_tile_pass(
        &self,
        _tile: &Tile,
        _refs: &RefList,
        _anchor_px: (i64, i64),
        _pass: u8,
        _iterations: usize,
        _int: &MultiInterrupter,
    ) -> bool {
        unimplemented!("wgpu perturbation backend (step 2)")
    }
}
