//! Tile rendering: progressive coarse-to-fine refinement of the tiles
//! visible in the current viewport, using perturbation theory against a
//! shared pool of reference orbits.

use crate::rendering::{
    CoordinatesBox, Orbit, Pixels, calculate_orbit, check_divergence_delta, check_orbit,
    val_to_color,
};
use crate::support::ToFBig;
use crate::support::append_only::List as AOList;
use crate::support::Point;
use crate::tiles::store::{
    NUM_PASSES, PASS_STRIDES, TILE_SIZE, Tile, TileKey, TileStore, depth_for_view, tile_index,
    units_per_pixel, units_per_pixel_fbig,
};
use dashu::float::FBig;
use dashu::integer::IBig;
use itertools::iproduct;
use log::{info, trace};
use num::Complex;
use ordered_float::OrderedFloat;
use rayon::ThreadPool;
use std::num::NonZeroUsize;
use std::sync::Arc;
#[cfg(debug_assertions)]
use std::sync::atomic::AtomicUsize;
#[cfg(debug_assertions)]
use std::sync::atomic::Ordering;
use waker_interrupter::MultiInterrupter;

/// How far (in current-view screen pixels) the viewport center may drift from
/// the orbit pool's anchor before the pool is rebuilt. Deltas are f64; with
/// ~16 significant digits, 1e7 pixels of drift still leaves plenty of
/// precision per pixel.
const MAX_ANCHOR_DISTANCE_PX: f64 = 1e7;

struct RefOrbit {
    /// delta of this orbit's base point from the pool anchor
    delta_corr: Complex<f64>,
    orbit: Orbit<f64>,
    #[cfg(debug_assertions)]
    hits: AtomicUsize,
}

/// A pool of reference orbits shared by all tiles. Deltas are measured from
/// `anchor`. Orbits are appended as rendering discovers points whose deltas
/// can't be tracked with the existing references (same condition as before:
/// `check_divergence_delta` fails because the reference escaped too early);
/// an orbit computed for a pixel in one tile is freely reused by other tiles.
pub struct OrbitPool {
    anchor: Complex<FBig>,
    iterations: usize,
    orbits: AOList<RefOrbit>,
}

impl OrbitPool {
    fn new(anchor: Complex<FBig>, iterations: usize) -> Self {
        let (_, orbit_f64) = calculate_orbit(anchor.clone(), iterations);
        let orbits = AOList::new();
        orbits.push_front(RefOrbit {
            delta_corr: Complex::ZERO,
            orbit: orbit_f64,
            #[cfg(debug_assertions)]
            hits: Default::default(),
        });
        Self {
            anchor,
            iterations,
            orbits,
        }
    }
}

/// Reuse the existing orbit pool if it still fits the viewport (same
/// iteration count, anchor close enough for f64 deltas), otherwise rebuild
/// it anchored at the viewport center.
fn ensure_pool(
    pool: &mut Option<OrbitPool>,
    coords: &CoordinatesBox,
    width: usize,
    height: usize,
    iterations: usize,
) {
    let view = coords.view.inner;
    let center = Complex {
        re: &coords.origin.x + &(width as f64 / 2.0 * view).to_fbig(),
        im: &coords.origin.y + &(height as f64 / 2.0 * view).to_fbig(),
    };

    if let Some(p) = pool {
        if p.iterations == iterations {
            let dx = (&p.anchor.re - &center.re).to_f64().value() / view;
            let dy = (&p.anchor.im - &center.im).to_f64().value() / view;
            if dx.hypot(dy) < MAX_ANCHOR_DISTANCE_PX {
                trace!("reusing orbit pool");
                return;
            }
        }
    }

    trace!("rebuilding orbit pool");
    *pool = Some(OrbitPool::new(center, iterations));
}

/// Everything fixed about a tile for the duration of a render pass.
struct TileCtx<'a> {
    tile: &'a Tile,
    /// tile origin - pool anchor, in units (f64 is fine: it's small)
    base_delta: Complex<f64>,
    /// units per tile pixel
    upp: f64,
    /// exact tile origin and units per pixel, for computing fresh reference
    /// orbits at full precision
    origin: Point<FBig, crate::rendering::Units>,
    upp_fbig: FBig,
}

/// Render one pixel of a tile (or skip it if a previous, interrupted run
/// already computed it). Returns whether the pixel is black (non-divergent).
fn render_pixel(ctx: &TileCtx, c: usize, r: usize, iterations: usize, pool: &OrbitPool) -> bool {
    let idx = r * TILE_SIZE + c;

    if let Some(raw) = ctx.tile.load(idx).get() {
        // already computed by an earlier (interrupted) run
        return raw == 0;
    }

    let delta = ctx.base_delta
        + Complex {
            re: c as f64 * ctx.upp,
            im: r as f64 * ctx.upp,
        };

    let val: Option<NonZeroUsize> = pool
        .orbits
        .iter()
        .find_map(|ref_orbit| {
            let out = check_divergence_delta(&ref_orbit.orbit, delta - ref_orbit.delta_corr).ok();

            #[cfg(debug_assertions)]
            if out.is_some() {
                ref_orbit.hits.fetch_add(1, Ordering::Relaxed);
            }

            out
        })
        .unwrap_or_else(|| {
            // no usable reference orbit: compute this point's own orbit at
            // full precision and add it to the pool as a new reference
            let x_0 = Complex {
                re: &ctx.origin.x + &(FBig::from_parts(IBig::from(c), 0) * &ctx.upp_fbig),
                im: &ctx.origin.y + &(FBig::from_parts(IBig::from(r), 0) * &ctx.upp_fbig),
            };

            let (_, new_orbit) = calculate_orbit(x_0, iterations);

            // guaranteed to be Ok(_)
            let val = check_orbit(&new_orbit).unwrap();

            pool.orbits.push_front(RefOrbit {
                delta_corr: delta,
                orbit: new_orbit,
                #[cfg(debug_assertions)]
                hits: Default::default(),
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
    pass: u8,
    iterations: usize,
    pool: &OrbitPool,
    int: &MultiInterrupter,
) -> bool {
    let stride = PASS_STRIDES[pass as usize];
    let coarser_stride = (pass > 0).then(|| PASS_STRIDES[pass as usize - 1]);

    let origin = tile.key.origin();
    let ctx = TileCtx {
        tile,
        base_delta: Complex {
            re: (&origin.x - &pool.anchor.re).to_f64().value(),
            im: (&origin.y - &pool.anchor.im).to_f64().value(),
        },
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
            all_black &= render_pixel(&ctx, c, r, iterations, pool);
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
                    perimeter_black &= render_pixel(&ctx, c, r, iterations, pool);
                }
            } else {
                for c in [0, TILE_SIZE - 1] {
                    perimeter_black &= render_pixel(&ctx, c, r, iterations, pool);
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
    pool_slot: &mut Option<OrbitPool>,
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

    ensure_pool(pool_slot, coords, width, height, iterations);
    let pool: &OrbitPool = pool_slot.as_ref().unwrap();

    // enumerate the tiles intersecting the viewport
    let x0 = tile_index(&coords.origin.x, depth);
    let y0 = tile_index(&coords.origin.y, depth);
    // screen-space size of a tile and position of tile (x0, y0)
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
        tile.render_gen.store(generation, std::sync::atomic::Ordering::Relaxed);

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
        tp.scope(|s| {
            for _ in 0..tp.current_num_threads() {
                s.spawn(move |_| {
                    while let Ok(tile) = rx.try_recv() {
                        if int.interrupted() {
                            return;
                        }
                        if render_tile_pass(&tile, pass, iterations, pool, int) {
                            store.bump_progress();
                        }
                    }
                });
            }
        });
    }

    store.bump_progress();

    #[cfg(debug_assertions)]
    {
        println!("{:>10} {:>10}", "length", "hits");
        for o in pool.orbits.iter() {
            println!(
                "{:>10} {:>10}",
                o.orbit.orbit.len(),
                o.hits.load(Ordering::Relaxed)
            );
        }
    }
}
