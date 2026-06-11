//! Tile rendering: progressive coarse-to-fine refinement of the tiles
//! visible in the current viewport, using perturbation theory.
//!
//! Tiles are bundled into large square *groups* (GROUP_TILES x GROUP_TILES
//! tiles, i.e. screen-sized or larger). All tiles in a group share one
//! reference-orbit list — the original whole-screen concurrent list, just
//! keyed per group instead of per frame. The list is the lock-free
//! append-only list, so the tiles of a group can render in parallel and
//! promote new references into the shared list concurrently, exactly as the
//! single-screen renderer did. A single reference covering a whole group is
//! fine: the original used a single reference for the entire screen at any
//! zoom, so the perturbation accuracy across a group is not a concern.
//!
//! Deltas are measured from the group's center (the anchor), in the same
//! spirit as before; because a tile's offset within its group is a small
//! integer, the per-pixel delta is computed in exact integer pixel units and
//! only then scaled, so it stays precise at any zoom.

use crate::rendering::{
    CoordinatesBox, Orbit, Pixels, calculate_orbit, check_divergence_delta, check_orbit,
    val_to_color,
};
use crate::support::Point;
use crate::support::append_only::List as AOList;
use crate::tiles::store::{
    NUM_PASSES, PASS_STRIDES, TILE_SIZE, Tile, TileKey, TileStore, depth_for_view, floor_div_pow2,
    tile_index, units_per_pixel, units_per_pixel_fbig,
};
use dashu::float::FBig;
use dashu::integer::IBig;
use itertools::iproduct;
use log::info;
use num::Complex;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use waker_interrupter::MultiInterrupter;

/// group side length, in tiles (2^GROUP_POW). 32 tiles => 4096 px per side,
/// comfortably larger than a screen, so the screen usually falls in 1-4
/// groups and only that many full-precision anchor orbits are computed.
const GROUP_POW: usize = 5;
const GROUP_TILES: usize = 1 << GROUP_POW;
/// pixel offset of the group center (anchor) from the group's top-left
const HALF_GROUP_PX: i64 = (GROUP_TILES * TILE_SIZE / 2) as i64;

/// drop the memoized group lists once the cache grows past this many groups,
/// to keep memory bounded
const GROUP_CACHE_CAP: usize = 256;

struct RefOrbit {
    /// this reference's base point, as a delta from the group anchor
    delta_corr: Complex<f64>,
    orbit: Orbit<f64>,
}

/// A group's shared reference list (the original lock-free concurrent list).
type GroupList = AOList<RefOrbit>;

#[derive(Clone, PartialEq, Eq, Hash)]
struct GroupKey {
    depth: i64,
    gx: IBig,
    gy: IBig,
}

fn group_of(key: &TileKey) -> GroupKey {
    GroupKey {
        depth: key.depth,
        gx: floor_div_pow2(&key.x, GROUP_POW),
        gy: floor_div_pow2(&key.y, GROUP_POW),
    }
}

/// Memoizes one shared list per group, seeded with the orbit of the group
/// center (anchor). Dropped when the iteration count changes.
pub struct GroupCache {
    iterations: usize,
    map: HashMap<GroupKey, GroupList>,
}

impl GroupCache {
    pub fn new() -> Self {
        Self {
            iterations: 0,
            map: HashMap::new(),
        }
    }
}

/// Get (or create and memoize) a group's shared reference list, seeded with
/// the orbit of the group center computed at full precision.
fn group_list(cache: &Mutex<GroupCache>, gkey: &GroupKey, iterations: usize) -> GroupList {
    {
        let c = cache.lock();
        if c.iterations == iterations {
            if let Some(list) = c.map.get(gkey) {
                return list.clone();
            }
        }
    }

    // compute the anchor (group center) orbit outside the lock
    let upp = units_per_pixel_fbig(gkey.depth);
    let group_origin = TileKey {
        depth: gkey.depth,
        x: &gkey.gx * IBig::from(GROUP_TILES as u64),
        y: &gkey.gy * IBig::from(GROUP_TILES as u64),
    }
    .origin();
    let half = FBig::from_parts(IBig::from(HALF_GROUP_PX), 0) * &upp;
    let anchor = Complex {
        re: &group_origin.x + &half,
        im: &group_origin.y + &half,
    };
    let (_, orbit) = calculate_orbit(anchor, iterations);

    let list: GroupList = AOList::new();
    list.push_front(RefOrbit {
        delta_corr: Complex::ZERO,
        orbit,
    });

    let mut c = cache.lock();
    if c.iterations != iterations {
        c.iterations = iterations;
        c.map.clear();
    }
    if c.map.len() >= GROUP_CACHE_CAP {
        c.map.clear();
    }
    c.map.entry(gkey.clone()).or_insert(list).clone()
}

/// Everything fixed about a tile for the duration of a render pass.
struct TileCtx<'a> {
    tile: &'a Tile,
    refs: &'a GroupList,
    /// units per tile pixel
    upp: f64,
    /// pixel offset of this tile's (0, 0) pixel from the group anchor, in the
    /// depth's exact integer pixel grid (small: |.| < group size)
    anchor_dx: i64,
    anchor_dy: i64,
    /// exact tile origin and units per pixel, for computing fresh reference
    /// orbits at full precision when a pixel promotes its own orbit
    origin: Point<FBig, crate::rendering::Units>,
    upp_fbig: FBig,
}

/// Render one pixel of a tile (or skip it if a previous, interrupted run
/// already computed it). Returns whether the pixel is black (non-divergent).
fn render_pixel(ctx: &TileCtx, c: usize, r: usize, iterations: usize) -> bool {
    let idx = r * TILE_SIZE + c;

    if let Some(raw) = ctx.tile.load(idx).get() {
        // already computed by an earlier (interrupted) run
        return raw == 0;
    }

    // delta of this pixel from the group anchor, computed in exact integer
    // pixel units first (so no precision is lost at deep zoom) then scaled
    let delta = Complex {
        re: (ctx.anchor_dx + c as i64) as f64 * ctx.upp,
        im: (ctx.anchor_dy + r as i64) as f64 * ctx.upp,
    };

    // try existing references, newest first (the list is push_front ordered)
    let val: Option<NonZeroUsize> = ctx
        .refs
        .iter()
        .find_map(|ref_orbit| {
            check_divergence_delta(&ref_orbit.orbit, delta - ref_orbit.delta_corr).ok()
        })
        .unwrap_or_else(|| {
            // no usable reference: compute this pixel's own orbit at full
            // precision and add it to the shared list so later pixels (in any
            // tile of the group) reuse it
            let x_0 = Complex {
                re: &ctx.origin.x + &(FBig::from_parts(IBig::from(c as i64), 0) * &ctx.upp_fbig),
                im: &ctx.origin.y + &(FBig::from_parts(IBig::from(r as i64), 0) * &ctx.upp_fbig),
            };
            let (_, new_orbit) = calculate_orbit(x_0, iterations);
            // guaranteed Ok(_): new_orbit is the true orbit of this point
            let val = check_orbit(&new_orbit).unwrap();

            ctx.refs.push_front(RefOrbit {
                delta_corr: delta,
                orbit: new_orbit,
            });

            val
        });

    ctx.tile.store(idx, val_to_color(val).into());
    val.is_none()
}

/// Run one progressive pass over a tile. Returns true if the pass ran to
/// completion (false = interrupted; partial pixels stay in the tile and are
/// skipped when the pass is retried).
fn render_tile_pass(
    tile: &Tile,
    refs: &GroupList,
    pass: u8,
    iterations: usize,
    int: &MultiInterrupter,
) -> bool {
    let stride = PASS_STRIDES[pass as usize];
    let coarser_stride = (pass > 0).then(|| PASS_STRIDES[pass as usize - 1]);

    // tile offset within its group (small integer), and the pixel offset of
    // this tile's origin from the group anchor
    let gkey = group_of(&tile.key);
    let local_x = i64::try_from(&(&tile.key.x - &gkey.gx * IBig::from(GROUP_TILES as u64)))
        .expect("tile-in-group offset fits i64");
    let local_y = i64::try_from(&(&tile.key.y - &gkey.gy * IBig::from(GROUP_TILES as u64)))
        .expect("tile-in-group offset fits i64");

    let origin = tile.key.origin();
    let ctx = TileCtx {
        tile,
        refs,
        upp: units_per_pixel(tile.key.depth),
        anchor_dx: local_x * TILE_SIZE as i64 - HALF_GROUP_PX,
        anchor_dy: local_y * TILE_SIZE as i64 - HALF_GROUP_PX,
        upp_fbig: units_per_pixel_fbig(tile.key.depth),
        origin,
    };

    let mut all_black = true;
    for r in (0..TILE_SIZE).step_by(stride) {
        if int.interrupted() {
            return false;
        }
        for c in (0..TILE_SIZE).step_by(stride) {
            if let Some(cs) = coarser_stride {
                if r % cs == 0 && c % cs == 0 {
                    // already computed by a coarser pass
                    continue;
                }
            }
            all_black &= render_pixel(&ctx, c, r, iterations);
        }
    }

    // Black-fill optimization: black (non-divergent) pixels are the most
    // expensive to compute. If the entire coarse pass-0 grid is black, render
    // the full-resolution perimeter; if that is black too, assume the whole
    // tile is black and fill it in one go.
    if pass == 0 && all_black {
        let mut perimeter_black = true;
        for r in 0..TILE_SIZE {
            if int.interrupted() {
                return false;
            }
            if r == 0 || r == TILE_SIZE - 1 {
                for c in 0..TILE_SIZE {
                    perimeter_black &= render_pixel(&ctx, c, r, iterations);
                }
            } else {
                for c in [0, TILE_SIZE - 1] {
                    perimeter_black &= render_pixel(&ctx, c, r, iterations);
                }
            }
        }

        if perimeter_black {
            for (r, c) in iproduct!(1..TILE_SIZE - 1, 1..TILE_SIZE - 1) {
                let idx = r * TILE_SIZE + c;
                if tile.load(idx).get().is_none() {
                    tile.store(idx, 0.into());
                }
            }
            tile.finish_pass(NUM_PASSES);
            return true;
        }
    }

    tile.finish_pass(pass + 1);
    true
}

/// Render all tiles visible in `coords`, coarse-to-fine across the whole
/// viewport: every visible tile gets pass 0 (sorted by distance from the
/// cursor/center), then every tile gets pass 1, and so on. Returns early
/// when interrupted by a newer request.
pub fn run_generation(
    store: &TileStore,
    group_cache: &Mutex<GroupCache>,
    width: usize,
    height: usize,
    coords: &CoordinatesBox,
    iterations: usize,
    cursor: Option<Point<usize, Pixels>>,
    int: MultiInterrupter,
    tp: &ThreadPool,
) {
    let generation = store.begin_generation();
    let view = coords.view.inner;
    let depth = depth_for_view(view);

    info!("render generation {generation}: depth {depth}, iterations {iterations}");

    // enumerate the tiles intersecting the viewport
    let x0 = tile_index(&coords.origin.x, depth);
    let y0 = tile_index(&coords.origin.y, depth);
    let tile_px = TILE_SIZE as f64 * (units_per_pixel(depth) / view);
    let origin00 = TileKey {
        depth,
        x: x0.clone(),
        y: y0.clone(),
    }
    .origin();
    let sx0 = (&origin00.x - &coords.origin.x).to_f64().value() / view;
    let sy0 = (&origin00.y - &coords.origin.y).to_f64().value() / view;
    let nx = ((width as f64 - sx0) / tile_px).ceil().max(1.0) as i64;
    let ny = ((height as f64 - sy0) / tile_px).ceil().max(1.0) as i64;

    let center = cursor.unwrap_or(Point::new(width / 2, height / 2));

    let mut tiles: Vec<(OrderedFloat<f64>, Arc<Tile>)> = vec![];
    for (i, j) in iproduct!(0..nx, 0..ny) {
        let key = TileKey {
            depth,
            x: &x0 + IBig::from(i),
            y: &y0 + IBig::from(j),
        };
        let tile = store.get_or_insert(&key, iterations, generation);
        tile.render_gen
            .store(generation, std::sync::atomic::Ordering::Relaxed);

        if tile.iterations.load(std::sync::atomic::Ordering::Relaxed) != iterations {
            // cached contents were computed with a different iteration count
            tile.reset(iterations);
        }
        if tile.is_complete() {
            // cache hit, nothing to do
            continue;
        }

        let cx = sx0 + (i as f64 + 0.5) * tile_px - center.x as f64;
        let cy = sy0 + (j as f64 + 0.5) * tile_px - center.y as f64;
        tiles.push((OrderedFloat(cx * cx + cy * cy), tile));
    }
    tiles.sort_unstable_by_key(|(dist, _)| *dist);

    store.evict_excess();

    info!("rendering {} tiles", tiles.len());

    // progressive passes: pass p starts only after every tile finished pass
    // p-1, so the whole canvas sharpens uniformly
    for pass in 0..NUM_PASSES {
        if int.interrupted() {
            return;
        }

        let (tx, rx) = crossbeam_channel::unbounded();
        tiles
            .iter()
            .filter(|(_, tile)| tile.passes_done() <= pass)
            .try_for_each(|(_, tile)| tx.send(tile.clone()))
            .expect("crossbeam channel failed");
        drop(tx);

        let rx = &rx;
        let int = &int;
        let group_cache = &*group_cache;
        tp.scope(|s| {
            for _ in 0..tp.current_num_threads() {
                s.spawn(move |_| {
                    while let Ok(tile) = rx.try_recv() {
                        if int.interrupted() {
                            return;
                        }
                        let refs = group_list(group_cache, &group_of(&tile.key), iterations);
                        if render_tile_pass(&tile, &refs, pass, iterations, int) {
                            store.bump_progress();
                        }
                    }
                });
            }
        });
    }

    store.bump_progress();
}
