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
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};

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

/// The pass that computes pixel (row, col) of a tile (inverse of
/// `pass_pixels`).
pub fn pass_of(r: usize, c: usize) -> u8 {
    if let Some(p) = GRID_STRIDES.iter().position(|&s| r.is_multiple_of(s) && c.is_multiple_of(s)) {
        return p as u8;
    }
    let parity = (r % 2, c % 2);
    NUM_GRID_PASSES + SUB_PASS_PARITY.iter().position(|&q| q == parity).unwrap() as u8
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

/// Minimum memory budget for the tile cache; old tiles are dropped (LRU)
/// beyond the budget, which grows with the view (`set_view_tiles`).
pub const MEMORY_BUDGET_BYTES: usize = 1 << 30; // 1 GiB ~ 16k tiles

/// The tile budget is this many times the current view's tiles (at least
/// the floor, at most the memory ceiling): room for their ancestors (+1/3,
/// for previews and seeding) and for recent views to go back to. E.g. at
/// s = 4 a 3000x2000 window needs up to ~24k tiles.
const VIEW_BUDGET_FACTOR: usize = 3;
const TILE_BYTES: usize = TILE_LEN * size_of::<u32>();

/// Memory per cached tile, counting every copy: the CPU store's pixels
/// (TILE_BYTES), the compositor's iterations on the GPU (as many) and its
/// colour levels (at most levels 0 and 1: 1.25×).
const TILE_TOTAL_BYTES: usize = TILE_BYTES * 13 / 4;

/// Sampling ratio s (←/→): steps and range.
pub const SAMPLING_STEP: f64 = 0.5;
pub const SAMPLING_MIN:  f64 = 0.5;
pub const SAMPLING_MAX:  f64 = 4.0;

/// Memory ceiling for the tile cache (all copies, see TILE_TOTAL_BYTES):
/// 3 GiB on wasm (a 4 GiB address space). None natively: the user wants
/// s = 4 at any window size, whatever it takes.
pub fn memory_ceiling_bytes() -> Option<usize> {
    cfg!(target_arch = "wasm32").then_some(3 << 30)
}

/// Worst-case number of tiles covering a `w`×`h` px window at sampling
/// ratio s: a screen pixel spans up to 2s tile pixels per axis (just before
/// a depth switch), plus a partial tile at each edge.
pub fn view_tiles_worst(s: f64, w: usize, h: usize) -> usize {
    let axis = |px: usize| (px as f64 * 2.0 * s / TILE_SIZE as f64).ceil() as usize + 1;
    axis(w) * axis(h)
}

/// The sampling ratio actually used: the largest s ≤ `requested`, in
/// SAMPLING_STEP steps, whose view fits `ceiling_tiles` even in the worst
/// case (so it doesn't flip while zooming); SAMPLING_MIN if none does. The
/// view's tiles are never evicted, so a view over the ceiling can't be
/// helped by the budget; before s drops, the cache around the view (the
/// VIEW_BUDGET_FACTOR) shrinks towards 1× (`set_view_tiles`).
pub fn effective_ratio(requested: f64, w: usize, h: usize, ceiling_tiles: usize) -> f64 {
    let mut s = requested;
    while s > SAMPLING_MIN && view_tiles_worst(s, w, h) > ceiling_tiles {
        s -= SAMPLING_STEP;
    }
    s.max(SAMPLING_MIN)
}

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
/// With a sampling ratio s, `TileStore::depth_for_view` passes `view / s`.
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
///
/// Pixels hold the escape iteration, not a colour: 0 = in the set, n =
/// escaped at iteration n (colouring happens in the compositor). That keeps
/// the contents meaningful when the iteration count changes (see `retarget`).
pub struct Tile {
    pub key: TileKey,
    pixels: UnsafeCell<Box<[u32]>>,
    /// number of fully completed progressive passes (0..=NUM_PASSES)
    passes_done: AtomicU8,
    /// passes to *display* regardless of `passes_done`: after raising the
    /// iteration count, the recomputed passes still show at their old
    /// resolution, with the pixels being recomputed drawn as in-set (black),
    /// which is what they were
    display_floor: AtomicU8,
    /// bumped whenever displayable contents change, so the compositor knows
    /// to re-upload
    version: AtomicU32,
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
            display_floor: AtomicU8::new(0),
            version: AtomicU32::new(0),
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
        self.version.fetch_add(1, Ordering::Release);
    }

    /// passes the compositor should display (see `display_floor`)
    pub fn display_passes(&self) -> u8 {
        self.passes_done().max(self.display_floor.load(Ordering::Acquire))
    }

    /// changes whenever the displayable contents change
    pub fn version(&self) -> u32 {
        self.version.load(Ordering::Acquire)
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

    /// Switch to a new iteration count, keeping every pixel that is still
    /// exact under it (no guesses are kept):
    /// - fewer iterations: a pixel that escaped after the new maximum is now
    ///   in the set; everything else is unchanged. Nothing to recompute.
    /// - more iterations: escaped pixels are unchanged; in-set pixels might
    ///   escape later, so they are cleared, and `passes_done` drops to the
    ///   passes still fully computed. The backends then recompute only the
    ///   cleared pixels, while the tile keeps displaying at its old level.
    ///
    /// Must not run while a backend is writing to this tile.
    pub fn retarget(&self, iterations: usize) {
        let old = self.iterations.swap(iterations, Ordering::Relaxed);
        if iterations == old { return; }
        let pixels = unsafe { &mut *self.pixels.get() };
        if iterations < old {
            let max = iterations as u32;
            for px in pixels.iter_mut() {
                if MaybePixel::from_raw(*px).get().is_some_and(|n| n > max) {
                    *px = MaybePixel::from(0).to_raw();
                }
            }
        } else {
            for px in pixels.iter_mut() {
                if MaybePixel::from_raw(*px).get() == Some(0) {
                    *px = MaybePixel::NONE_RAW;
                }
            }
            let shown = self.display_passes();
            let complete = (0..NUM_PASSES)
                .take_while(|&p| pass_pixels(p).all(|(r, c)| MaybePixel::from_raw(pixels[r * TILE_SIZE + c]).get().is_some()))
                .count() as u8;
            self.passes_done.store(complete.min(self.passes_done()), Ordering::Release);
            self.display_floor.store(shown, Ordering::Release);
        }
        self.version.fetch_add(1, Ordering::Release);
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
    /// Floor of the tile budget (`MEMORY_BUDGET_BYTES` in tiles).
    min_tiles: usize,
    /// Ceiling of the tile budget (`memory_ceiling_bytes` in tiles;
    /// usize::MAX: none).
    ceiling_tiles: usize,
    /// Current tile budget: VIEW_BUDGET_FACTOR × view tiles, within
    /// [min_tiles, ceiling_tiles].
    max_tiles: AtomicUsize,
    /// tiles of the current view (`set_view_tiles`)
    view_tiles: AtomicUsize,
    /// Sampling ratio s as requested (←/→), as f32 bits.
    requested_ratio: AtomicU32,
    /// Sampling ratio in use (`effective_ratio`): the smallest number of
    /// tile pixels per screen pixel (per axis) to render at, as f32 bits;
    /// see `depth_for_view`.
    min_ratio: AtomicU32,
    /// window size in physical pixels (w << 32 | h), for `effective_ratio`
    window: AtomicU64,
}

impl TileStore {
    pub fn new(memory_budget_bytes: usize) -> Self {
        Self::with_ceiling(memory_budget_bytes, memory_ceiling_bytes())
    }

    /// A store with the memory ceiling `ceiling_bytes` (None: none).
    pub fn with_ceiling(memory_budget_bytes: usize, ceiling_bytes: Option<usize>) -> Self {
        Self {
            map: FlurryMap::new(),
            frame: AtomicU64::new(0),
            progress: Arc::new(AtomicU64::new(0)),
            generation: AtomicU64::new(0),
            reset: AtomicBool::new(false),
            min_tiles: (memory_budget_bytes / TILE_BYTES).max(64),
            ceiling_tiles: ceiling_bytes.map_or(usize::MAX, |b| (b / TILE_TOTAL_BYTES).max(64)),
            max_tiles: AtomicUsize::new((memory_budget_bytes / TILE_BYTES).max(64)),
            view_tiles: AtomicUsize::new(0),
            requested_ratio: AtomicU32::new(1.0f32.to_bits()),
            min_ratio: AtomicU32::new(1.0f32.to_bits()),
            window: AtomicU64::new(0),
        }
    }

    /// Sampling ratio s: tiles are rendered at the depth where one screen
    /// pixel spans between s and 2s tile pixels per axis (s = 1: between
    /// 1:1 and 2:1; the compositor averages them down to the screen).
    /// This is the effective s (see `effective_ratio`), which is below
    /// `requested_ratio` when that doesn't fit the memory ceiling.
    pub fn min_ratio(&self) -> f64 {
        f32::from_bits(self.min_ratio.load(Ordering::Relaxed)) as f64
    }

    /// The sampling ratio set with ←/→.
    pub fn requested_ratio(&self) -> f64 {
        f32::from_bits(self.requested_ratio.load(Ordering::Relaxed)) as f64
    }

    pub fn set_requested_ratio(&self, s: f64) {
        self.requested_ratio.store((s as f32).to_bits(), Ordering::Relaxed);
        self.update_ratio(true);
    }

    /// The window size changed: an s that fitted may no longer (or again).
    pub fn set_window(&self, w: usize, h: usize) {
        self.window.store(((w as u64) << 32) | h as u64, Ordering::Relaxed);
        self.update_ratio(false);
    }

    /// Recompute the effective s. Logged when it changes, or (after an s
    /// change, `requested`) whenever it is clamped: never clamped silently.
    fn update_ratio(&self, requested: bool) {
        let window = self.window.load(Ordering::Relaxed);
        let (w, h) = ((window >> 32) as usize, (window & 0xFFFF_FFFF) as usize);
        let want = self.requested_ratio();
        let s = effective_ratio(want, w, h, self.ceiling_tiles);
        let old = f32::from_bits(self.min_ratio.swap((s as f32).to_bits(), Ordering::Relaxed)) as f64;
        if s < want && (s != old || requested) {
            log::warn!(
                "sampling: s = {want} needs up to {} tiles for a {w}x{h} window, over the memory \
                 ceiling of {} tiles ({:.1} GiB); using s = {s}",
                view_tiles_worst(want, w, h), self.ceiling_tiles,
                (self.ceiling_tiles * TILE_TOTAL_BYTES) as f64 / (1u64 << 30) as f64,
            );
        } else if s != old {
            log::info!("sampling: s = {s} fits a {w}x{h} window again");
        }
    }

    /// The tile budget over the current view's tiles: VIEW_BUDGET_FACTOR,
    /// unless the memory ceiling cut it (None before the first view).
    pub fn cache_factor(&self) -> Option<f64> {
        let view = self.view_tiles.load(Ordering::Relaxed);
        (view > 0).then(|| self.max_tiles.load(Ordering::Relaxed) as f64 / view as f64)
    }

    /// The depth to render a view at (units per screen pixel) with the
    /// current sampling ratio.
    pub fn depth_for_view(&self, view: f64) -> i64 {
        depth_for_view(view / self.min_ratio())
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

    /// Size the budget for a view of `n` tiles: it grows with the view (a
    /// high sampling ratio needs many more tiles) and shrinks back when the
    /// view needs fewer, so memory is released by the next `evict_excess`.
    /// Capped by the memory ceiling (never below the view itself, which is
    /// never evicted; `effective_ratio` keeps it under the ceiling).
    pub fn set_view_tiles(&self, n: usize) {
        let budget = self.min_tiles.max(VIEW_BUDGET_FACTOR * n).min(self.ceiling_tiles.max(n));
        self.max_tiles.store(budget, Ordering::Relaxed);
        self.view_tiles.store(n, Ordering::Relaxed);
    }

    /// Drop least-recently-displayed tiles until the cache fits the budget.
    /// Tiles belonging to the current render generation are never dropped.
    pub fn evict_excess(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        let guard = self.map.guard();
        let max_tiles = self.max_tiles.load(Ordering::Relaxed);
        if self.map.len() <= max_tiles {
            return;
        }
        let excess = self.map.len() - max_tiles;

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

    /// The budget follows the view: a large view raises it (so its ancestors
    /// and recent views are kept), a small one lowers it again and the next
    /// eviction releases the excess, never the current generation's tiles.
    #[test]
    fn budget_follows_the_view() {
        let store = TileStore::new(64 * TILE_BYTES); // floor: 64 tiles
        let key = |d: i64, i: i64| TileKey { depth: d, x: IBig::from(i), y: IBig::ZERO };
        let generation = store.begin_generation();
        for i in 0..100 { store.get_or_insert(&key(3, i), 100, generation); }
        store.set_view_tiles(100);
        store.evict_excess();
        assert_eq!(store.map.len(), 100); // budget 300: nothing to drop

        let generation = store.begin_generation();
        for i in 0..10 {
            store.get_or_insert(&key(5, i), 100, generation).render_gen.store(generation, Ordering::Relaxed);
        }
        store.set_view_tiles(10); // budget back to the 64-tile floor
        store.evict_excess();
        assert_eq!(store.map.len(), 64);
        for i in 0..10 { assert!(store.get(&key(5, i)).is_some(), "current tile {i} evicted"); }
    }

    /// The effective s is the requested one while the worst-case view fits
    /// the ceiling, else the largest step that does (never below the
    /// minimum); shrinking the window gives the requested s back.
    #[test]
    fn effective_ratio_fits_the_ceiling() {
        let (w, h) = (3000, 2000);
        // s = 4: (ceil(3000·8/128) + 1) × (ceil(2000·8/128) + 1)
        assert_eq!(view_tiles_worst(4.0, w, h), 189 * 126);
        assert_eq!(effective_ratio(4.0, w, h, 1 << 30), 4.0);
        assert_eq!(effective_ratio(4.0, w, h, 189 * 126), 4.0);
        assert_eq!(effective_ratio(4.0, w, h, 189 * 126 - 1), 3.5);
        let fits_3 = view_tiles_worst(3.0, w, h);
        assert_eq!(effective_ratio(4.0, w, h, fits_3), 3.0);
        assert_eq!(effective_ratio(4.0, w, h, 1), SAMPLING_MIN);
        assert_eq!(effective_ratio(4.0, w / 2, h / 2, fits_3), 4.0);
        assert_eq!(effective_ratio(1.0, w, h, fits_3), 1.0);
    }

    /// The store applies it on window and s changes, and caps the budget
    /// (a wasm-like 3 GiB ceiling); natively there is none.
    #[test]
    fn store_clamps_s_to_the_ceiling() {
        let native = TileStore::new(1 << 20);
        native.set_window(100_000, 100_000);
        native.set_requested_ratio(4.0);
        assert_eq!(native.min_ratio(), 4.0);
        native.set_view_tiles(1 << 20);
        assert_eq!(native.cache_factor(), Some(VIEW_BUDGET_FACTOR as f64));

        let store = TileStore::with_ceiling(1 << 20, Some(3 << 30));
        let ceiling = store.ceiling_tiles;
        // a window so large that s = 4 can't fit (but s = 0.5 can)
        let side = ((ceiling as f64).sqrt() * TILE_SIZE as f64 / 4.0) as usize;
        store.set_window(side, side);
        store.set_requested_ratio(4.0);
        assert_eq!(store.requested_ratio(), 4.0);
        assert!(store.min_ratio() < 4.0, "effective {}", store.min_ratio());
        assert!(view_tiles_worst(store.min_ratio(), side, side) <= ceiling);
        store.set_window(100, 100);
        assert_eq!(store.min_ratio(), 4.0);
        store.set_view_tiles(ceiling);
        assert_eq!(store.cache_factor(), Some(1.0));
        store.set_view_tiles(10);
        assert!(store.cache_factor().unwrap() >= VIEW_BUDGET_FACTOR as f64);
    }

    /// With sampling ratio s, a screen pixel spans between s and 2s tile
    /// pixels per axis.
    #[test]
    fn depth_follows_sampling_ratio() {
        let store = TileStore::new(1 << 20);
        for s in [0.5, 1.0, 1.5, 2.0, 2.5, 4.0] {
            store.set_requested_ratio(s);
            for i in 0..200 {
                let view = (-30.0 + i as f64 * 0.137).exp2();
                let ratio = view / units_per_pixel(store.depth_for_view(view));
                assert!(ratio >= s && ratio < 2.0 * s, "s {s}, view {view:e}: ratio {ratio}");
            }
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
    fn pass_of_inverts_pass_pixels() {
        for pass in 0..NUM_PASSES {
            for (r, c) in pass_pixels(pass) {
                assert_eq!(pass_of(r, c), pass, "({r},{c})");
            }
        }
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

    /// What a backend stores for a pixel with true escape time `e` under
    /// `iterations`: e if it escapes in time, else 0 (in the set).
    fn computed(e: u32, iterations: usize) -> u32 {
        if e as usize <= iterations { e } else { 0 }
    }

    /// Retargeting keeps only exact values: retarget N1 -> N2, recompute
    /// just the cleared pixels, and the tile equals a direct N2 render.
    #[test]
    fn retarget_matches_direct_render() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        for _ in 0..40 {
            let truth: Vec<u32> = (0..TILE_LEN).map(|_| 1 + (rand() % 5000) as u32).collect();
            let n1 = 256 << (rand() % 5);
            let n2 = 256 << (rand() % 5);
            let store = TileStore::new(1 << 26);
            let tile = store.get_or_insert(&TileKey { depth: 1, x: IBig::from(0), y: IBig::from(0) }, n1, 0);
            for (i, &e) in truth.iter().enumerate() { tile.store(i, computed(e, n1).into()); }
            tile.finish_pass(NUM_PASSES);
            let v0 = tile.version();

            tile.retarget(n2);
            assert!(tile.version() != v0 || n1 == n2);
            assert_eq!(tile.display_passes(), NUM_PASSES, "keeps displaying");
            if n2 <= n1 {
                assert!(tile.is_complete(), "fewer iterations never needs recomputing");
            }
            let mut recomputed = 0;
            for (i, &e) in truth.iter().enumerate() {
                if tile.load(i).get().is_none() {
                    tile.store(i, computed(e, n2).into());
                    recomputed += 1;
                }
                assert_eq!(tile.load(i).get(), Some(computed(e, n2)), "pixel {i}, {n1} -> {n2}");
            }
            // Only formerly in-set pixels are recomputed.
            let in_set_before = truth.iter().filter(|&&e| computed(e, n1) == 0).count();
            assert_eq!(recomputed, if n2 > n1 { in_set_before } else { 0 });
        }
    }

    /// After raising iterations, passes_done drops to the leading passes
    /// whose pixels are all still known.
    #[test]
    fn retarget_up_lowers_passes_to_complete_prefix() {
        let store = TileStore::new(1 << 26);
        let tile = store.get_or_insert(&TileKey { depth: 1, x: IBig::from(0), y: IBig::from(0) }, 100, 0);
        for i in 0..TILE_LEN { tile.store(i, 50.into()); }
        // One in-set pixel in the first sub-pass.
        let (r, c) = pass_pixels(NUM_GRID_PASSES).next().unwrap();
        tile.store(r * TILE_SIZE + c, 0.into());
        tile.finish_pass(NUM_PASSES);
        tile.retarget(200);
        assert_eq!(tile.passes_done(), NUM_GRID_PASSES);
        assert_eq!(tile.display_passes(), NUM_PASSES);
    }
}
