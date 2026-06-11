//! Tile rendering: progressive coarse-to-fine refinement of the tiles
//! visible in the current viewport, using perturbation theory.
//!
//! Each tile owns its own reference-orbit list (the original algorithm,
//! scoped to a tile instead of shared globally). The list starts with the
//! orbit of the tile center and grows as pixels that can't be tracked
//! against an existing reference promote their own orbit into it — so a
//! single expensive (e.g. long, non-diverging) orbit computed once is reused
//! by every other pixel in the tile, instead of being recomputed per pixel.
//!
//! Because each tile is rendered start-to-finish by a single worker (one
//! tile per pass goes to one thread, and passes run sequentially behind a
//! barrier), the list is only ever touched by one thread at a time. The
//! promotion order is therefore fixed and rendering is deterministic: a given
//! tile (at a given iteration count) always renders to exactly the same
//! pixels, which is what keeps the image stable when zooming back and forth.

use crate::rendering::{
    CoordinatesBox, Orbit, Pixels, calculate_orbit, check_divergence_delta, check_orbit,
    val_to_color,
};
use crate::support::Point;
use crate::support::append_only::List as AOList;
use crate::tiles::store::{
    NUM_PASSES, PASS_STRIDES, TILE_SIZE, Tile, TileKey, TileStore, depth_for_view, tile_index,
    units_per_pixel, units_per_pixel_fbig,
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

/// the pixel coordinate the tile's first reference orbit is computed at
/// (tile center); deltas are measured from here, so they stay small
const REF_PIXEL: f64 = (TILE_SIZE / 2) as f64;

/// drop the memoized reference lists once the cache grows past this many
/// tiles, to keep memory bounded
const REF_CACHE_CAP: usize = 2048;

struct RefOrbit {
    /// this reference's base point, as a delta from the tile center
    delta_corr: Complex<f64>,
    orbit: Orbit<f64>,
}

/// Memoizes one reference-orbit list per tile. The list is a pure function of
/// the tile (same center orbit, same deterministic promotion order), so
/// memoizing it is a speed-up that also keeps the chosen references stable
/// across re-renders. The whole cache is dropped when the iteration count
/// changes.
pub struct RefCache {
    iterations: usize,
    map: HashMap<TileKey, AOList<RefOrbit>>,
}

impl RefCache {
    pub fn new() -> Self {
        Self {
            iterations: 0,
            map: HashMap::new(),
        }
    }
}

/// Get (or create and memoize) a tile's reference list, seeded with the orbit
/// of the tile center computed at full precision.
fn tile_ref_list(cache: &Mutex<RefCache>, key: &TileKey, iterations: usize) -> AOList<RefOrbit> {
    {
        let c = cache.lock();
        if c.iterations == iterations {
            if let Some(list) = c.map.get(key) {
                return list.clone();
            }
        }
    }

    // compute the center orbit outside the lock (the expensive part)
    let upp = units_per_pixel_fbig(key.depth);
    let origin = key.origin();
    let ref_offset = FBig::from_parts(IBig::from(REF_PIXEL as i64), 0) * &upp;
    let center = Complex {
        re: &origin.x + &ref_offset,
        im: &origin.y + &ref_offset,
    };
    let (_, orbit) = calculate_orbit(center, iterations);

    let list = AOList::new();
    list.push_front(RefOrbit {
        delta_corr: Complex::ZERO,
        orbit,
    });

    let mut c = cache.lock();
    if c.iterations != iterations {
        c.iterations = iterations;
        c.map.clear();
    }
    if c.map.len() >= REF_CACHE_CAP {
        c.map.clear();
    }
    c.map.entry(key.clone()).or_insert(list).clone()
}

/// Everything fixed about a tile for the duration of a render pass.
struct TileCtx<'a> {
    tile: &'a Tile,
    refs: &'a AOList<RefOrbit>,
    /// units per tile pixel
    upp: f64,
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

    // delta of this pixel from the tile center (the first reference's base
    // point); small and bounded (< ~90 px * upp), so f64 perturbation is
    // accurate
    let delta = Complex {
        re: (c as f64 - REF_PIXEL) * ctx.upp,
        im: (r as f64 - REF_PIXEL) * ctx.upp,
    };

    let val: Option<NonZeroUsize> = ctx
        .refs
        .iter()
        .find_map(|ref_orbit| {
            check_divergence_delta(&ref_orbit.orbit, delta - ref_orbit.delta_corr).ok()
        })
        .unwrap_or_else(|| {
            // no usable reference: compute this pixel's own orbit at full
            // precision and add it to the list so later pixels reuse it
            // (this is what stops a short reference + many long orbits from
            // recomputing a full orbit per pixel)
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
    refs: &AOList<RefOrbit>,
    pass: u8,
    iterations: usize,
    int: &MultiInterrupter,
) -> bool {
    let stride = PASS_STRIDES[pass as usize];
    let coarser_stride = (pass > 0).then(|| PASS_STRIDES[pass as usize - 1]);

    let origin = tile.key.origin();
    let ctx = TileCtx {
        tile,
        refs,
        upp: units_per_pixel(tile.key.depth),
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
    ref_cache: &Mutex<RefCache>,
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
        let ref_cache = &*ref_cache;
        tp.scope(|s| {
            for _ in 0..tp.current_num_threads() {
                s.spawn(move |_| {
                    while let Ok(tile) = rx.try_recv() {
                        if int.interrupted() {
                            return;
                        }
                        let refs = tile_ref_list(ref_cache, &tile.key, iterations);
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
