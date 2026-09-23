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
use flurry::HashMap as FlurryMap;
use std::sync::Arc;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};

pub const TILE_POW: usize = 7;
/// side length of a tile in pixels
pub const TILE_SIZE: usize = 1 << TILE_POW;
pub const TILE_LEN: usize = TILE_SIZE * TILE_SIZE;

/// log2 of the span (in units) of a depth-0 tile
pub const DEPTH_0_TILE_SPAN_LOG2: i64 = 2; // 4 units

/// Strides of the square-grid refinement passes. Pass 0 renders a coarse
/// grid of every 16th pixel, each subsequent grid pass fills in the pixels
/// needed to halve the stride. After grid pass p the grid of every
/// GRID_STRIDES[p]-th pixel is fully computed, so the compositor can
/// interpolate between those pixels and refinement looks like the resolution
/// increasing.
pub const GRID_STRIDES: [usize; 4] = [16, 8, 4, 2];
pub const NUM_GRID_PASSES: u8 = GRID_STRIDES.len() as u8;

/// The last step, stride 2 → 1, is 75% of a tile's pixels, so it is split
/// into three equal sub-passes over the stride-2 cells, given as (row, col)
/// parity. The cell centres come first: that completes a quincunx lattice in
/// which every remaining pixel has all four axis neighbours computed, so the
/// compositor can fill it from them. Each later sub-pass keeps that property.
const SUB_PASS_PARITY: [(usize, usize); 3] = [(1, 1), (0, 1), (1, 0)];

pub const NUM_PASSES: u8 = NUM_GRID_PASSES + SUB_PASS_PARITY.len() as u8;

/// Passes done once the grid of every `stride`-th pixel is complete (0 if
/// that grid is coarser than pass 0's).
pub fn passes_for_stride(stride: usize) -> u8 {
    if stride == 1 {
        NUM_PASSES
    } else {
        GRID_STRIDES.iter().filter(|&&s| s >= stride).count() as u8
    }
}

/// (row, col) of the pixels that `pass` adds to a tile, in row-major order.
pub fn pass_pixels(pass: u8) -> impl Iterator<Item = (usize, usize)> {
    let p = pass as usize;
    let (stride, coarser, (r0, c0)) = if p < GRID_STRIDES.len() {
        (GRID_STRIDES[p], p.checked_sub(1).map(|q| GRID_STRIDES[q]), (0, 0))
    } else {
        (2, None, SUB_PASS_PARITY[p - GRID_STRIDES.len()])
    };
    (r0..TILE_SIZE).step_by(stride)
        .flat_map(move |r| (c0..TILE_SIZE).step_by(stride).map(move |c| (r, c)))
        // skip pixels a coarser grid pass already did
        .filter(move |&(r, c)| coarser.is_none_or(|cs| r % cs != 0 || c % cs != 0))
}

/// tiles are bundled into GROUP_TILES x GROUP_TILES groups that share one
/// reference-orbit list
pub const GROUP_POW: usize = 4;
pub const GROUP_TILES: usize = 1 << GROUP_POW;

/// memory budget for the tile cache; old tiles are dropped (LRU) beyond this
pub const MEMORY_BUDGET_BYTES: usize = 1 << 30; // 1 GiB ~ 16k tiles
const TILE_BYTES: usize = TILE_LEN * size_of::<u32>();

/// log2 of units per pixel at a given depth
pub fn upp_log2(depth: i64) -> i64 {
    DEPTH_0_TILE_SPAN_LOG2 - TILE_POW as i64 - depth
}

/// units per pixel at a given depth, as f64 (subject to f64 range limits,
/// same as `View`)
pub fn units_per_pixel(depth: i64) -> f64 {
    (upp_log2(depth) as f64).exp2()
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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
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

}

/// A single rendered (or partially rendered) tile.
pub struct Tile {
    pub key: TileKey,
    pixels: UnsafeCell<Box<[u32]>>,
    /// number of fully completed progressive passes (0..=NUM_PASSES)
    passes_done: AtomicU8,
    /// iteration count the contents were/are being computed with
    pub iterations: AtomicUsize,
    /// frame counter value when this tile last contributed to the display
    pub last_used: AtomicU64,
    /// render generation that last scheduled this tile (eviction protection)
    pub render_gen: AtomicU64,
}

// SAFETY: pixels are written by render threads and read by the compositor
// concurrently. u32 reads/writes are naturally atomic on x86/ARM; we accept
// the theoretical data race in exchange for zero synchronization overhead.
unsafe impl Sync for Tile {}

impl Tile {
    fn new(key: TileKey, iterations: usize, render_gen: u64) -> Self {
        Self {
            key,
            pixels: UnsafeCell::new(vec![MaybePixel::NONE_RAW; TILE_LEN].into_boxed_slice()),
            passes_done: AtomicU8::new(0),
            iterations: AtomicUsize::new(iterations),
            last_used: AtomicU64::new(0),
            render_gen: AtomicU64::new(render_gen),
        }
    }

    pub fn load(&self, idx: usize) -> MaybePixel {
        MaybePixel::from_raw(unsafe { (*self.pixels.get())[idx] })
    }

    pub fn store(&self, idx: usize, px: MaybePixel) {
        unsafe { (*self.pixels.get())[idx] = px.to_raw() }
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
    /// (during the sub-passes the stride-2 grid is complete, plus some of the
    /// stride-1 pixels)
    pub fn completed_stride(&self) -> Option<usize> {
        match self.passes_done() {
            0 => None,
            p if p >= NUM_PASSES => Some(1),
            p => Some(GRID_STRIDES[(p.min(NUM_GRID_PASSES) - 1) as usize]),
        }
    }

    /// wipe contents for re-rendering with a different iteration count
    pub fn reset(&self, iterations: usize) {
        self.passes_done.store(0, Ordering::Release);
        self.iterations.store(iterations, Ordering::Relaxed);
        unsafe { (*self.pixels.get()).fill(MaybePixel::NONE_RAW) };
    }

    pub fn touch(&self, frame: u64) {
        self.last_used.fetch_max(frame, Ordering::Relaxed);
    }
}

/// Concurrent tile cache with a memory budget and LRU eviction.
pub struct TileStore {
    map: FlurryMap<TileKey, Arc<Tile>>,
    /// display frame counter, bumped by the compositor (drives LRU)
    frame: AtomicU64,
    /// bumped whenever rendered pixels become displayable (tells the drawer
    /// to recompose); shared with backends so they can report partial passes
    progress: Arc<AtomicU64>,
    /// current render generation
    generation: AtomicU64,
    /// TEST: set by spacebar; the next render dumps everything and restarts
    reset: AtomicBool,
    max_tiles: usize,
}

impl TileStore {
    pub fn new(memory_budget_bytes: usize) -> Self {
        Self {
            map: FlurryMap::new(),
            frame: AtomicU64::new(0),
            progress: Arc::new(AtomicU64::new(0)),
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
        let guard = self.map.guard();
        self.map.retain(|_, _| false, &guard);
        self.bump_progress();
    }

    pub fn get(&self, key: &TileKey) -> Option<Arc<Tile>> {
        self.map.get(key, &self.map.guard()).cloned()
    }

    pub fn get_or_insert(&self, key: &TileKey, iterations: usize, render_gen: u64) -> Arc<Tile> {
        let guard = self.map.guard();
        if let Some(tile) = self.map.get(key, &guard) {
            return tile.clone();
        }
        let tile = Arc::new(Tile::new(key.clone(), iterations, render_gen));
        self.map.insert(key.clone(), tile.clone(), &guard);
        tile
    }

    /// Seed `tile` from already-rendered tiles one level up or down. Their
    /// pixel grids nest exactly: pixel (i, j) at depth d is the same point as
    /// pixel (2i, 2j) at depth d+1, since `upp_log2` drops by one per level.
    ///
    /// - Parent → child: the parent's quadrant fills the child's even-even
    ///   pixels, so a parent complete at stride s completes the child's
    ///   stride-2s grid (a finished parent gives passes 0–3 for free).
    /// - Children → parent: each child's even-even pixels fill one quadrant,
    ///   so four children complete at stride s complete the parent at s/2.
    ///
    /// Only tiles computed with the same iteration count are used, and only
    /// when that raises `tile`'s completed passes. Returns whether it did.
    /// Must not run while a backend is writing to these tiles.
    pub fn seed_from_relatives(&self, tile: &Tile) -> bool {
        let iterations = tile.iterations.load(Ordering::Relaxed);
        let usable = |t: &Arc<Tile>| {
            (t.iterations.load(Ordering::Relaxed) == iterations).then(|| t.completed_stride()).flatten()
        };
        let half = TILE_SIZE / 2;
        let copy_quadrant = |from: &Tile, to: &Tile, parent_to_child: bool, (bx, by): (usize, usize)| {
            for i in 0..half {
                for j in 0..half {
                    let parent_idx = (by * half + i) * TILE_SIZE + bx * half + j;
                    let child_idx  = (2 * i) * TILE_SIZE + 2 * j;
                    let (src, dst) = if parent_to_child { (parent_idx, child_idx) } else { (child_idx, parent_idx) };
                    if to.load(dst).get().is_none() && from.load(src).get().is_some() {
                        to.store(dst, from.load(src));
                    }
                }
            }
        };

        // Parent → child.
        let parent_level = self.get(&tile.key.parent())
            .and_then(|p| usable(&p).map(|s| (p, passes_for_stride(2 * s))))
            .filter(|(_, level)| *level > tile.passes_done());
        // Children → parent (needs all four).
        let children: Option<Vec<_>> = [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().map(|(bx, by)| {
            let key = TileKey {
                depth: tile.key.depth + 1,
                x: &tile.key.x * IBig::from(2) + IBig::from(bx),
                y: &tile.key.y * IBig::from(2) + IBig::from(by),
            };
            let child = self.get(&key)?;
            let stride = usable(&child)?;
            Some((child, (bx as usize, by as usize), stride))
        }).collect();
        let children_level = children.as_ref()
            .map(|cs| passes_for_stride(cs.iter().map(|&(_, _, s)| (s / 2).max(1)).max().unwrap()))
            .filter(|level| *level > tile.passes_done());

        let mut level = 0;
        if let Some((parent, l)) = &parent_level {
            let (bx, by) = tile.key.parent_offset();
            copy_quadrant(parent, tile, true, (bx as usize, by as usize));
            level = level.max(*l);
        }
        if let (Some(cs), Some(l)) = (&children, children_level) {
            for (child, quadrant, _) in cs {
                copy_quadrant(child, tile, false, *quadrant);
            }
            level = level.max(l);
        }
        if level > tile.passes_done() {
            tile.finish_pass(level);
            true
        } else {
            false
        }
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

    /// Shared handle on the progress counter, for backends that finish
    /// tiles part-way through a pass.
    pub fn progress_counter(&self) -> Arc<AtomicU64> {
        self.progress.clone()
    }

    pub fn next_frame(&self) -> u64 {
        self.frame.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Drop least-recently-displayed tiles until the cache fits the budget.
    /// Tiles belonging to the current render generation are never dropped.
    pub fn evict_excess(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        let guard = self.map.guard();
        if self.map.len() <= self.max_tiles {
            return;
        }
        let excess = self.map.len() - self.max_tiles;

        let mut candidates: Vec<(u64, TileKey)> = self.map
            .iter(&guard)
            .filter(|(_, t)| t.render_gen.load(Ordering::Relaxed) != generation)
            .map(|(k, t)| (t.last_used.load(Ordering::Relaxed), k.clone()))
            .collect();
        candidates.sort_unstable_by_key(|(last_used, _)| *last_used);

        for (_, key) in candidates.into_iter().take(excess) {
            self.map.remove(&key, &guard);
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
        // -5 = 2 * (-3) + 1
        assert_eq!(key.parent().x, IBig::from(-3));
        assert_eq!(bx, 1);
        assert_eq!(by, 1);
    }
}

#[cfg(test)]
mod pass_tests {
    use super::*;

    #[test]
    fn passes_cover_every_pixel_once() {
        let mut seen = vec![0u8; TILE_LEN];
        for pass in 0..NUM_PASSES {
            for (r, c) in pass_pixels(pass) {
                seen[r * TILE_SIZE + c] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1));
    }

    #[test]
    fn pass_sizes() {
        let sizes: Vec<usize> = (0..NUM_PASSES).map(|p| pass_pixels(p).count()).collect();
        assert_eq!(sizes, [64, 192, 768, 3072, 4096, 4096, 4096]);
    }

    /// What `GpuCompositor::reconstruct` relies on: once any sub-pass is
    /// done, every missing pixel has all its in-tile axis neighbours.
    #[test]
    fn sub_passes_leave_only_fully_surrounded_gaps() {
        for done in NUM_GRID_PASSES + 1..NUM_PASSES {
            let mut known = vec![false; TILE_LEN];
            for pass in 0..done {
                for (r, c) in pass_pixels(pass) { known[r * TILE_SIZE + c] = true; }
            }
            for r in 0..TILE_SIZE {
                for c in 0..TILE_SIZE {
                    if known[r * TILE_SIZE + c] { continue; }
                    let n = TILE_SIZE as isize;
                    for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let (rr, cc) = (r as isize + dr, c as isize + dc);
                        if (0..n).contains(&rr) && (0..n).contains(&cc) {
                            assert!(known[(rr * n + cc) as usize], "pass {done}: ({r},{c}) lacks ({rr},{cc})");
                        }
                    }
                }
            }
        }
    }

    /// Unique non-black value per pixel, remembering where it came from.
    fn fill(tile: &Tile, id: u32) {
        for idx in 0..TILE_LEN {
            tile.store(idx, MaybePixel::from((id << 16) | idx as u32)); // id >= 1: never black
        }
        tile.finish_pass(NUM_PASSES);
    }

    fn global_coord(key: &TileKey, idx: usize) -> (FBig, FBig) {
        let ts = IBig::from(TILE_SIZE as u64);
        let (r, c) = (idx / TILE_SIZE, idx % TILE_SIZE);
        (
            pixel_to_coord(&key.x * &ts + IBig::from(c), key.depth, 64),
            pixel_to_coord(&key.y * &ts + IBig::from(r), key.depth, 64),
        )
    }

    /// Every seeded pixel must be the *same point* as the pixel it was
    /// copied from, for negative and positive tile indices alike.
    #[test]
    fn seeding_copies_identical_points() {
        for (x, y) in [(5i64, -3i64), (-7, 4), (0, 0), (-1, -1)] {
            // Parent -> child, for each of the four children.
            let store = TileStore::new(1 << 26);
            let parent_key = TileKey { depth: 10, x: IBig::from(x), y: IBig::from(y) };
            fill(&store.get_or_insert(&parent_key, 100, 0), 1);
            for (bx, by) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let key = TileKey { depth: 11, x: IBig::from(2 * x + bx), y: IBig::from(2 * y + by) };
                assert_eq!(key.parent(), parent_key);
                let child = store.get_or_insert(&key, 100, 0);
                assert!(store.seed_from_relatives(&child));
                assert_eq!(child.passes_done(), NUM_GRID_PASSES); // stride-2 grid
                let mut copied = 0;
                for idx in 0..TILE_LEN {
                    let Some(v) = child.load(idx).get() else { continue };
                    let src = (v & 0xFFFF) as usize;
                    assert_eq!(global_coord(&key, idx), global_coord(&parent_key, src));
                    copied += 1;
                }
                assert_eq!(copied, TILE_LEN / 4);
            }

            // Children -> parent.
            let store = TileStore::new(1 << 26);
            let mut keys = vec![];
            for (i, (bx, by)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                let key = TileKey { depth: 11, x: IBig::from(2 * x + bx), y: IBig::from(2 * y + by) };
                fill(&store.get_or_insert(&key, 100, 0), i as u32 + 1);
                keys.push(key);
            }
            let parent = store.get_or_insert(&parent_key, 100, 0);
            assert!(store.seed_from_relatives(&parent));
            assert!(parent.is_complete());
            for idx in 0..TILE_LEN {
                let v = parent.load(idx).get().expect("parent fully seeded");
                let child_key = &keys[(v >> 16) as usize - 1];
                assert_eq!(global_coord(&parent_key, idx), global_coord(child_key, (v & 0xFFFF) as usize));
            }
        }
    }

    #[test]
    fn seeding_needs_same_iterations() {
        let store = TileStore::new(1 << 26);
        let parent_key = TileKey { depth: 3, x: IBig::from(1), y: IBig::from(1) };
        fill(&store.get_or_insert(&parent_key, 100, 0), 1);
        let child = store.get_or_insert(&TileKey { depth: 4, x: IBig::from(2), y: IBig::from(3) }, 200, 0);
        assert!(!store.seed_from_relatives(&child));
        assert_eq!(child.passes_done(), 0);
    }

    #[test]
    fn stride_to_passes() {
        assert_eq!([32, 16, 8, 4, 2, 1].map(passes_for_stride), [0, 1, 2, 3, 4, NUM_PASSES]);
    }
}
