//! Tile cache: the canvas (really the complex plane) is split into a quadtree
//! of square tiles. A tile is identified by an integer depth and an integer
//! (x, y) index pair; the tile covers the square
//!   [x * span(depth), (x + 1) * span(depth)) x [y * span(depth), (y + 1) * span(depth))
//! in complex-plane units, where span(depth) = 2^(DEPTH_0_TILE_SPAN_LOG2 - depth).
//!
//! Tiles are fixed to the plane, not the screen: once rendered they stay valid
//! forever (for a given iteration count), so zooming and panning never throw
//! work away. Each deeper level halves the tile span, i.e. doubles resolution.

use crate::drawing::maybe_pixel::MaybePixel;
use crate::rendering::Units;
use crate::support::Point;
use dashu::float::FBig;
use dashu::integer::IBig;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};

pub const TILE_POW: usize = 7;
/// side length of a tile in pixels
pub const TILE_SIZE: usize = 1 << TILE_POW;
pub const TILE_LEN: usize = TILE_SIZE * TILE_SIZE;

/// log2 of the span (in units) of a depth-0 tile
pub const DEPTH_0_TILE_SPAN_LOG2: i64 = 2; // 4 units

/// Strides of the progressive refinement passes. Pass 0 renders a coarse
/// grid of every 16th pixel, each subsequent pass fills in the pixels needed
/// to halve the stride. After pass p the grid of every PASS_STRIDES[p]-th
/// pixel is fully computed, so the compositor can interpolate between those
/// pixels and refinement looks like the resolution increasing.
pub const PASS_STRIDES: [usize; 5] = [16, 8, 4, 2, 1];
pub const NUM_PASSES: u8 = PASS_STRIDES.len() as u8;

/// tiles are bundled into GROUP_TILES x GROUP_TILES groups that share one
/// reference-orbit list
pub const GROUP_POW: usize = 4;
pub const GROUP_TILES: usize = 1 << GROUP_POW;

/// memory budget for the tile cache; old tiles are dropped (LRU) beyond this
pub const MEMORY_BUDGET_BYTES: usize = 1 << 30; // 1 GiB ~ 16k tiles
const TILE_BYTES: usize = TILE_LEN * size_of::<u32>();

/// 2^e as an exact FBig (works far outside the f64 exponent range)
pub fn pow2(e: i64) -> FBig {
    FBig::from_parts(IBig::from(1), 0) << e as isize
}

/// log2 of units per pixel at a given depth
fn upp_log2(depth: i64) -> i64 {
    DEPTH_0_TILE_SPAN_LOG2 - TILE_POW as i64 - depth
}

/// units per pixel at a given depth, as f64 (subject to f64 range limits,
/// same as `View`)
pub fn units_per_pixel(depth: i64) -> f64 {
    (upp_log2(depth) as f64).exp2()
}

/// units per pixel at a given depth, exact
pub fn units_per_pixel_fbig(depth: i64) -> FBig {
    pow2(upp_log2(depth))
}

/// Working precision (bits) for reference-orbit base points at a given depth.
///
/// CRITICAL: FBig infers its precision from how it is constructed
/// (`from_parts` uses the significand's bit length, and arithmetic rounds to
/// `Context::max` of the operands). If we build a reference point from a small
/// integer tile index it ends up with only a handful of bits of precision, so
/// the reference orbit — and therefore the whole perturbation — is computed
/// far less accurately than even f64. Reference points must instead be built
/// at a precision that comfortably exceeds f64, scaled with depth (deeper =
/// more bits to even locate the point).
pub fn working_precision(depth: i64) -> usize {
    depth.max(0) as usize + 128
}

/// Exact absolute coordinate of a pixel (in the depth's global pixel grid),
/// built at `prec` bits so the reference orbit computed from it is accurate.
/// The value is `pixel * 2^upp_log2(depth)`, exact, with a high-precision
/// context attached for the subsequent orbit arithmetic.
pub fn pixel_to_coord(pixel: IBig, depth: i64, prec: usize) -> FBig {
    FBig::from_parts(pixel, 0).with_precision(prec).value() << upp_log2(depth) as isize
}

/// The depth at which tiles should be rendered for a given view scale
/// (units per screen pixel): the smallest depth whose tile resolution is at
/// least the screen resolution, so tiles are displayed at scale (1/2, 1].
pub fn depth_for_view(view: f64) -> i64 {
    let d = (DEPTH_0_TILE_SPAN_LOG2 as f64 - TILE_POW as f64 - view.log2()).ceil();
    if d.is_nan() {
        0
    } else {
        // stay strictly inside the f64 exponent range
        (d as i64).clamp(-900, 1050)
    }
}

/// floor(i / 2)
pub fn floor_div2(i: &IBig) -> IBig {
    let q = i / IBig::from(2);
    // `/` truncates towards zero; fix up for negative odd numbers
    if &(&q * IBig::from(2)) > i {
        q - IBig::from(1)
    } else {
        q
    }
}

/// floor(i / 2^pow)
pub fn floor_div_pow2(i: &IBig, pow: usize) -> IBig {
    let d = IBig::from(1u64 << pow);
    let q = i / &d;
    // `/` truncates towards zero; fix up for negatives
    if &(&q * &d) > i {
        q - IBig::from(1)
    } else {
        q
    }
}

/// index of the tile containing `coord` at `depth`
///
/// NOTE: `coord` must carry enough precision to resolve the tile grid at
/// this depth (~depth + a margin of bits), otherwise the result is garbage.
/// `main` keeps the view origin's precision topped up as the zoom deepens.
pub fn tile_index(coord: &FBig, depth: i64) -> IBig {
    let span_log2 = upp_log2(depth) + TILE_POW as i64;
    (coord.clone() << (-span_log2) as isize).floor().to_int().value()
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TileKey {
    pub depth: i64,
    pub x: IBig,
    pub y: IBig,
}

impl TileKey {
    /// absolute coordinates of the top-left corner of this tile (exact)
    pub fn origin(&self) -> Point<FBig, Units> {
        let span_log2 = (upp_log2(self.depth) + TILE_POW as i64) as isize;
        Point::new(
            FBig::from_parts(self.x.clone(), 0) << span_log2,
            FBig::from_parts(self.y.clone(), 0) << span_log2,
        )
    }

    /// the tile one depth level up that contains this tile
    pub fn parent(&self) -> TileKey {
        TileKey {
            depth: self.depth - 1,
            x: floor_div2(&self.x),
            y: floor_div2(&self.y),
        }
    }

    /// `self`'s offset within its parent: (bit_x, bit_y), each 0 or 1
    pub fn parent_offset(&self) -> (u64, u64) {
        let two = IBig::from(2);
        let bx = &self.x - floor_div2(&self.x) * &two;
        let by = &self.y - floor_div2(&self.y) * &two;
        (
            (bx == IBig::from(1)) as u64,
            (by == IBig::from(1)) as u64,
        )
    }

    /// the child tile at sub-position (cx, cy), cx and cy each 0 or 1
    pub fn child(&self, cx: u64, cy: u64) -> TileKey {
        let two = IBig::from(2);
        TileKey {
            depth: self.depth + 1,
            x: &self.x * &two + IBig::from(cx),
            y: &self.y * &two + IBig::from(cy),
        }
    }
}

/// A single rendered (or partially rendered) tile.
///
/// Pixels are atomics so the render threads can fill the tile in while the
/// compositor reads it; a torn frame at worst shows a pixel one frame early.
pub struct Tile {
    pub key: TileKey,
    pixels: Box<[AtomicU32]>,
    /// number of fully completed progressive passes (0..=NUM_PASSES)
    passes_done: AtomicU8,
    /// iteration count the contents were/are being computed with
    pub iterations: AtomicUsize,
    /// frame counter value when this tile last contributed to the display
    pub last_used: AtomicU64,
    /// render generation that last scheduled this tile (eviction protection)
    pub render_gen: AtomicU64,
}

impl Tile {
    fn new(key: TileKey, iterations: usize, render_gen: u64) -> Self {
        Self {
            key,
            pixels: (0..TILE_LEN)
                .map(|_| AtomicU32::new(MaybePixel::NONE_RAW))
                .collect(),
            passes_done: AtomicU8::new(0),
            iterations: AtomicUsize::new(iterations),
            last_used: AtomicU64::new(0),
            render_gen: AtomicU64::new(render_gen),
        }
    }

    pub fn load(&self, idx: usize) -> MaybePixel {
        MaybePixel::from_raw(self.pixels[idx].load(Ordering::Relaxed))
    }

    pub fn store(&self, idx: usize, px: MaybePixel) {
        self.pixels[idx].store(px.to_raw(), Ordering::Relaxed);
    }

    pub fn passes_done(&self) -> u8 {
        self.passes_done.load(Ordering::Acquire)
    }

    /// monotonically raise the completed-passes counter
    pub fn finish_pass(&self, passes_done: u8) {
        self.passes_done.fetch_max(passes_done, Ordering::Release);
    }

    pub fn is_complete(&self) -> bool {
        self.passes_done() >= NUM_PASSES
    }

    /// the finest fully-computed grid stride, None if nothing is computed yet
    pub fn completed_stride(&self) -> Option<usize> {
        match self.passes_done() {
            0 => None,
            p => Some(PASS_STRIDES[(p - 1).min(NUM_PASSES - 1) as usize]),
        }
    }

    /// wipe contents for re-rendering with a different iteration count
    pub fn reset(&self, iterations: usize) {
        self.passes_done.store(0, Ordering::Release);
        self.iterations.store(iterations, Ordering::Relaxed);
        for p in &self.pixels {
            p.store(MaybePixel::NONE_RAW, Ordering::Relaxed);
        }
    }

    pub fn touch(&self, frame: u64) {
        self.last_used.fetch_max(frame, Ordering::Relaxed);
    }
}

/// Concurrent tile cache with a memory budget and LRU eviction.
pub struct TileStore {
    map: Mutex<HashMap<TileKey, Arc<Tile>>>,
    /// display frame counter, bumped by the compositor (drives LRU)
    frame: AtomicU64,
    /// bumped whenever a render pass completes (tells the drawer to recompose)
    progress: AtomicU64,
    /// current render generation
    generation: AtomicU64,
    /// TEST: set by spacebar; the next render dumps everything and restarts
    reset: AtomicBool,
    max_tiles: usize,
}

impl TileStore {
    pub fn new(memory_budget_bytes: usize) -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            frame: AtomicU64::new(0),
            progress: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            reset: AtomicBool::new(false),
            max_tiles: (memory_budget_bytes / TILE_BYTES).max(64),
        }
    }

    /// TEST: request a full reset (dump all tiles) on the next render.
    pub fn request_reset(&self) {
        self.reset.store(true, Ordering::Release);
    }

    /// TEST: consume the reset request.
    pub fn take_reset(&self) -> bool {
        self.reset.swap(false, Ordering::AcqRel)
    }

    /// TEST: drop every cached tile.
    pub fn clear(&self) {
        self.map.lock().clear();
        self.bump_progress();
    }

    pub fn get(&self, key: &TileKey) -> Option<Arc<Tile>> {
        self.map.lock().get(key).cloned()
    }

    pub fn get_or_insert(&self, key: &TileKey, iterations: usize, render_gen: u64) -> Arc<Tile> {
        self.map
            .lock()
            .entry(key.clone())
            .or_insert_with(|| Arc::new(Tile::new(key.clone(), iterations, render_gen)))
            .clone()
    }

    pub fn begin_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn bump_progress(&self) {
        self.progress.fetch_add(1, Ordering::Release);
    }

    pub fn progress(&self) -> u64 {
        self.progress.load(Ordering::Acquire)
    }

    pub fn next_frame(&self) -> u64 {
        self.frame.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Drop least-recently-displayed tiles until the cache fits the budget.
    /// Tiles belonging to the current render generation are never dropped.
    pub fn evict_excess(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        let mut map = self.map.lock();
        if map.len() <= self.max_tiles {
            return;
        }
        let excess = map.len() - self.max_tiles;

        let mut candidates: Vec<(u64, TileKey)> = map
            .values()
            .filter(|t| t.render_gen.load(Ordering::Relaxed) != generation)
            .map(|t| (t.last_used.load(Ordering::Relaxed), t.key.clone()))
            .collect();
        candidates.sort_unstable_by_key(|(last_used, _)| *last_used);

        for (_, key) in candidates.into_iter().take(excess) {
            map.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_div2() {
        for (i, exp) in [(5, 2), (4, 2), (1, 0), (0, 0), (-1, -1), (-4, -2), (-5, -3)] {
            assert_eq!(floor_div2(&IBig::from(i)), IBig::from(exp), "i = {i}");
        }
    }

    #[test]
    fn test_depth_for_view() {
        // at depth d, units per pixel = 2^(2 - 7 - d) = 2^(-5 - d)
        assert_eq!(depth_for_view(1.0 / 32.0), 0);
        assert_eq!(depth_for_view(1.0 / 300.0), 4); // 2^-9 <= 1/300 < 2^-8
        assert_eq!(depth_for_view(1.0 / 64.0), 1);
        // tile resolution is always >= screen resolution
        for view_log2 in -60..10 {
            let view = (view_log2 as f64).exp2() * 1.3;
            let d = depth_for_view(view);
            assert!(units_per_pixel(d) <= view, "view_log2 = {view_log2}");
            assert!(units_per_pixel(d) > view / 2.0, "view_log2 = {view_log2}");
        }
    }

    #[test]
    fn test_tile_index_and_origin() {
        let f = |x: f64| -> FBig { FBig::try_from(x).unwrap() };
        // depth 0 tile spans 4 units, tile 0 covers [0, 4)
        assert_eq!(tile_index(&f(3.9), 0), IBig::from(0));
        assert_eq!(tile_index(&f(4.0), 0), IBig::from(1));
        assert_eq!(tile_index(&f(-0.1), 0), IBig::from(-1));
        // depth 5 tile spans 4/32 = 0.125 units
        assert_eq!(tile_index(&f(0.25), 5), IBig::from(2));

        let key = TileKey {
            depth: 5,
            x: IBig::from(-3),
            y: IBig::from(2),
        };
        let origin = key.origin();
        assert_eq!(origin.x.to_f64().value(), -0.375);
        assert_eq!(origin.y.to_f64().value(), 0.25);

        // origin lies in the tile it indexes
        assert_eq!(tile_index(&origin.x, 5), IBig::from(-3));
    }

    #[test]
    fn test_parent_child_roundtrip() {
        let key = TileKey {
            depth: 3,
            x: IBig::from(-5),
            y: IBig::from(7),
        };
        let (bx, by) = key.parent_offset();
        assert_eq!(key.parent().child(bx, by), key);
        // -5 = 2 * (-3) + 1
        assert_eq!(key.parent().x, IBig::from(-3));
        assert_eq!(bx, 1);
        assert_eq!(by, 1);
    }

    #[test]
    fn test_pow2() {
        assert_eq!(pow2(3).to_f64().value(), 8.0);
        assert_eq!(pow2(-2).to_f64().value(), 0.25);
        // far outside the f64 range, must stay exact
        let tiny = pow2(-2000);
        assert_eq!((tiny << 2000).to_f64().value(), 1.0);
    }
}
