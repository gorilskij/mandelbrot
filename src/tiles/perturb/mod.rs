//! Perturbation backends.
//!
//! `tiles::render` enumerates the visible tiles, manages the per-group
//! reference lists, and drives the progressive passes.  For each pass it
//! hands the full set of tiles needing that pass to `Perturbator::render_pass_batch`.
//!
//! Two implementations:
//!   - [`cpu::Cpu`]  — f32 perturbation on the CPU, parallelised with rayon.
//!   - [`gpu::Gpu`]  — wgpu compute; batches all tiles in a pass into a single
//!     GPU dispatch with iterative glitch-correction passes.

pub mod bla;
pub mod cpu;
pub mod gpu;
pub mod nucleus;

use crate::rendering::{CoordinatesBox, Orbit, Pf};
use crate::support::append_only::List as AOList;
use crate::tiles::store::Tile;
use cpu::Cpu;
use dashu::integer::IBig;
use gpu::Gpu;
use num::Complex;
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use waker_interrupter::MultiInterrupter;

/// A reference orbit (projected to `Pf`) together with its base point's delta
/// from the group anchor.
pub struct RefOrbit {
    pub delta_corr: Complex<Pf>,
    pub orbit:      Orbit<Pf>,
}

/// A group's shared, growable reference list (lock-free append-only).
pub type RefList = AOList<RefOrbit>;

// ---------------------------------------------------------------------------
// Batch context
// ---------------------------------------------------------------------------

/// Per-pass context shared across all tiles in a `render_pass_batch` call.
pub struct PassBatchCtx {
    pub coords:     CoordinatesBox,
    /// Tile-grid depth.
    pub depth:      i64,
    /// Tile-grid origin (top-left tile index at this depth).
    pub x0:         IBig,
    pub y0:         IBig,
    pub width:      usize,
    pub height:     usize,
    pub iterations: usize,
    /// The store's progress counter: bump it after finishing tiles
    /// mid-pass so the compositor picks them up.
    pub progress:   Arc<AtomicU64>,
}

/// One tile's entry in a batch.
pub struct TileItem {
    pub tile:      Arc<Tile>,
    /// CPU uses this to look up existing reference orbits.
    pub refs:      RefList,
    /// CPU uses this — pixel offset of the group anchor within the group.
    pub anchor_px: (i64, i64),
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A backend's work on one batch. Async because on the web a worker can't
/// block on GPU readbacks; natively it is driven with a blocking executor
/// (`pollster`) and behaves as before. Not `Send`: on the web GPU objects
/// stay on the thread that made them.
pub type BatchFuture<'a> = Pin<Box<dyn Future<Output = ()> + 'a>>;

/// Renders a batch of tiles for one progressive pass.
///
/// The tile store is updated in place; `finish_pass` is called on each tile
/// upon completion (backends may skip it if interrupted mid-batch).
pub trait Perturbator: Sync {
    fn render_pass_batch<'a>(
        &'a self,
        ctx:   &'a PassBatchCtx,
        tiles: &'a [TileItem],
        pass:  u8,
        int:   &'a MultiInterrupter,
    ) -> BatchFuture<'a>;
}

// ---------------------------------------------------------------------------
// Toggle
// ---------------------------------------------------------------------------

/// Delegates to either [`Cpu`] or [`Gpu`] based on a shared atomic flag.
pub struct Toggle {
    pub cpu:     Cpu,
    pub gpu:     Gpu,
    pub use_gpu: Arc<AtomicBool>,
}

impl Toggle {
    pub fn new(gpu: Gpu) -> (Self, Arc<AtomicBool>) {
        let use_gpu = Arc::new(AtomicBool::new(true)); // start on the GPU backend
        let toggle = Toggle { cpu: Cpu, gpu, use_gpu: use_gpu.clone() };
        (toggle, use_gpu)
    }
}

impl Perturbator for Toggle {
    fn render_pass_batch<'a>(
        &'a self,
        ctx:   &'a PassBatchCtx,
        tiles: &'a [TileItem],
        pass:  u8,
        int:   &'a MultiInterrupter,
    ) -> BatchFuture<'a> {
        if self.use_gpu.load(Ordering::Relaxed) {
            self.gpu.render_pass_batch(ctx, tiles, pass, int)
        } else {
            self.cpu.render_pass_batch(ctx, tiles, pass, int)
        }
    }
}
