//! Tile rendering: progressive coarse-to-fine refinement of the tiles
//! visible in the current viewport, using perturbation theory.
//!
//! Tiles are bundled into large square *groups* (GROUP_TILES x GROUP_TILES
//! tiles). All tiles in a group share one reference-orbit list — the original
//! whole-screen concurrent (lock-free append-only) list, keyed per group. The
//! group's seed reference is the FBig orbit of a point in the group (its
//! center normally; a RANDOM point in TEST/spacebar mode, so different
//! reference choices can be compared). Other pixels are tracked as f64 deltas
//! against the list, promoting their own orbit when no reference fits.

use crate::rendering::{
    CoordinatesBox, Orbit, Pixels, calculate_orbit, check_divergence_delta, check_orbit,
    val_to_color,
};
use crate::support::Point;
use crate::support::append_only::List as AOList;
use crate::tiles::store::{
    GROUP_POW, GROUP_TILES, NUM_PASSES, PASS_STRIDES, TILE_SIZE, Tile, TileKey, TileStore,
    depth_for_view, floor_div_pow2, pixel_to_coord, tile_index, units_per_pixel, working_precision,
};
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

/// group side length in pixels (GROUP_TILES tiles * TILE_SIZE px)
const GROUP_SIDE_PX: i64 = (GROUP_TILES * TILE_SIZE) as i64;

/// drop the memoized group lists once the cache grows past this many groups
const GROUP_CACHE_CAP: usize = 256;

/// TEST: when true, each group's seed reference is a random point in the
/// group instead of its center (so pressing space re-rolls the references).
const RANDOM_REFERENCE: bool = false;

/// TEST: when true, render everything single-threaded in a fixed order
/// (tiles in sorted order, pixels in loop order) so the shared group list
/// grows in one deterministic sequence and output is reproducible.
const DETERMINISTIC: bool = false;

struct RefOrbit {
    /// this reference's base point, as a delta from the group anchor
    delta_corr: Complex<f64>,
    orbit: Orbit<f64>,
}

/// A group's shared reference list plus its anchor (the pixel offset, within
/// the group, that deltas are measured from — the seed reference's location).
#[derive(Clone)]
struct GroupRef {
    list: AOList<RefOrbit>,
    anchor_px: (i64, i64),
}

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

/// Memoizes one shared list per group. Dropped when the iteration count
/// changes or on an explicit reset (spacebar).
pub struct GroupCache {
    iterations: usize,
    map: HashMap<GroupKey, GroupRef>,
}

impl GroupCache {
    pub fn new() -> Self {
        Self {
            iterations: 0,
            map: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.iterations = 0;
        self.map.clear();
    }
}

/// Get (or create and memoize) a group's shared reference list, seeded with
/// the FBig orbit of the group's anchor point computed at full precision.
fn group_list(cache: &Mutex<GroupCache>, gkey: &GroupKey, iterations: usize) -> GroupRef {
    {
        let c = cache.lock();
        if c.iterations == iterations {
            if let Some(gref) = c.map.get(gkey) {
                return gref.clone();
            }
        }
    }

    // choose the anchor pixel offset within the group
    let anchor_px = if RANDOM_REFERENCE {
        let pick = || ((rand::random::<f64>() * GROUP_SIDE_PX as f64) as i64).clamp(0, GROUP_SIDE_PX - 1);
        (pick(), pick())
    } else {
        (GROUP_SIDE_PX / 2, GROUP_SIDE_PX / 2)
    };

    // anchor absolute coordinate, built at high precision (see
    // working_precision) so the reference orbit is actually accurate
    let prec = working_precision(gkey.depth);
    let group_px = IBig::from((GROUP_TILES * TILE_SIZE) as u64);
    let anchor_global_x = &gkey.gx * &group_px + IBig::from(anchor_px.0);
    let anchor_global_y = &gkey.gy * &group_px + IBig::from(anchor_px.1);
    let anchor = Complex {
        re: pixel_to_coord(anchor_global_x, gkey.depth, prec),
        im: pixel_to_coord(anchor_global_y, gkey.depth, prec),
    };
    let (_, orbit) = calculate_orbit(anchor, iterations);

    let list = AOList::new();
    list.push_front(RefOrbit {
        delta_corr: Complex::ZERO,
        orbit,
    });
    let gref = GroupRef { list, anchor_px };

    let mut c = cache.lock();
    if c.iterations != iterations {
        c.iterations = iterations;
        c.map.clear();
    }
    if c.map.len() >= GROUP_CACHE_CAP {
        c.map.clear();
    }
    c.map.entry(gkey.clone()).or_insert(gref).clone()
}

/// Everything fixed about a tile for the duration of a render pass.
struct TileCtx<'a> {
    tile: &'a Tile,
    refs: &'a AOList<RefOrbit>,
    /// units per tile pixel
    upp: f64,
    /// pixel offset of this tile's (0, 0) pixel from the group anchor, in the
    /// depth's exact integer pixel grid (small: |.| < group size)
    anchor_dx: i64,
    anchor_dy: i64,
    /// global pixel coordinate of this tile's (0, 0) pixel, and the depth /
    /// precision, for building fresh reference orbits at high precision when a
    /// pixel promotes its own orbit
    tile_px0_x: IBig,
    tile_px0_y: IBig,
    depth: i64,
    prec: usize,
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
            // no usable reference: compute this pixel's own orbit at high
            // precision and add it to the shared list so later pixels (in any
            // tile of the group) reuse it
            let x_0 = Complex {
                re: pixel_to_coord(&ctx.tile_px0_x + IBig::from(c as u64), ctx.depth, ctx.prec),
                im: pixel_to_coord(&ctx.tile_px0_y + IBig::from(r as u64), ctx.depth, ctx.prec),
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
    refs: &AOList<RefOrbit>,
    anchor_px: (i64, i64),
    pass: u8,
    iterations: usize,
    int: &MultiInterrupter,
) -> bool {
    let stride = PASS_STRIDES[pass as usize];
    let coarser_stride = (pass > 0).then(|| PASS_STRIDES[pass as usize - 1]);

    // tile offset within its group (small integer)
    let gkey = group_of(&tile.key);
    let local_x = i64::try_from(&(&tile.key.x - &gkey.gx * IBig::from(GROUP_TILES as u64)))
        .expect("tile-in-group offset fits i64");
    let local_y = i64::try_from(&(&tile.key.y - &gkey.gy * IBig::from(GROUP_TILES as u64)))
        .expect("tile-in-group offset fits i64");

    let depth = tile.key.depth;
    let tile_size = IBig::from(TILE_SIZE as u64);
    let ctx = TileCtx {
        tile,
        refs,
        upp: units_per_pixel(depth),
        anchor_dx: local_x * TILE_SIZE as i64 - anchor_px.0,
        anchor_dy: local_y * TILE_SIZE as i64 - anchor_px.1,
        tile_px0_x: &tile.key.x * &tile_size,
        tile_px0_y: &tile.key.y * &tile_size,
        depth,
        prec: working_precision(depth),
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
    // TEST: spacebar requests a full reset — dump every tile and every group
    // reference list, so everything is recomputed with fresh random references
    if store.take_reset() {
        store.clear();
        group_cache.lock().clear();
        info!("reset: dumped all tiles and reference lists");
    }

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

        if DETERMINISTIC {
            // TEST: single-threaded, fixed order — fully reproducible
            for (_, tile) in tiles.iter() {
                if int.interrupted() {
                    return;
                }
                if tile.passes_done() > pass {
                    continue;
                }
                let gref = group_list(group_cache, &group_of(&tile.key), iterations);
                if render_tile_pass(tile, &gref.list, gref.anchor_px, pass, iterations, &int) {
                    store.bump_progress();
                }
            }
            continue;
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
                        let gref = group_list(group_cache, &group_of(&tile.key), iterations);
                        if render_tile_pass(
                            &tile,
                            &gref.list,
                            gref.anchor_px,
                            pass,
                            iterations,
                            int,
                        ) {
                            store.bump_progress();
                        }
                    }
                });
            }
        });
    }

    store.bump_progress();
}
