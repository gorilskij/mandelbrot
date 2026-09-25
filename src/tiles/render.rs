//! Tile rendering orchestration: enumerate the tiles intersecting the
//! viewport, manage the per-group reference lists, and drive the progressive
//! passes. The actual per-pixel work is delegated to a `Perturbator` backend
//! via `render_pass_batch` — one call per pass with *all* tiles that need it.

use crate::rendering::{CoordinatesBox, Pixels, calculate_orbit};
use crate::support::Point;
use crate::tiles::perturb::{PassBatchCtx, Perturbator, RefList, RefOrbit, TileItem};
use crate::tiles::store::{
    GROUP_POW, GROUP_TILES, NUM_PASSES, TILE_SIZE, TileKey, TileStore,
    floor_div_pow2, pixel_to_coord, tile_index, units_per_pixel, working_precision,
};
use dashu::integer::IBig;
use itertools::iproduct;
use log::info;
use num::Complex;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use waker_interrupter::MultiInterrupter;

const GROUP_SIDE_PX: i64 = (GROUP_TILES * TILE_SIZE) as i64;
const GROUP_CACHE_CAP: usize = 256;

/// TEST: use a random point per group instead of the centre.
const RANDOM_REFERENCE: bool = false;

#[derive(Clone)]
struct GroupRef {
    list:      RefList,
    anchor_px: (i64, i64),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct GroupKey {
    depth: i64,
    gx:    IBig,
    gy:    IBig,
}

fn group_of(key: &TileKey) -> GroupKey {
    GroupKey {
        depth: key.depth,
        gx:    floor_div_pow2(&key.x, GROUP_POW),
        gy:    floor_div_pow2(&key.y, GROUP_POW),
    }
}

pub struct GroupCache {
    iterations: usize,
    map:        HashMap<GroupKey, GroupRef>,
}

impl GroupCache {
    pub fn new() -> Self { Self { iterations: 0, map: HashMap::new() } }
    pub fn clear(&mut self) { self.iterations = 0; self.map.clear(); }
}

fn group_list(cache: &Mutex<GroupCache>, gkey: &GroupKey, iterations: usize) -> GroupRef {
    {
        let c = cache.lock();
        if c.iterations == iterations
            && let Some(gref) = c.map.get(gkey) { return gref.clone(); }
    }

    let anchor_px = if RANDOM_REFERENCE {
        let pick = || ((rand::random::<f64>() * GROUP_SIDE_PX as f64) as i64)
            .clamp(0, GROUP_SIDE_PX - 1);
        (pick(), pick())
    } else {
        (GROUP_SIDE_PX / 2, GROUP_SIDE_PX / 2)
    };

    let prec       = working_precision(gkey.depth);
    let group_px   = IBig::from((GROUP_TILES * TILE_SIZE) as u64);
    let anchor     = Complex {
        re: pixel_to_coord(&gkey.gx * &group_px + IBig::from(anchor_px.0), gkey.depth, prec),
        im: pixel_to_coord(&gkey.gy * &group_px + IBig::from(anchor_px.1), gkey.depth, prec),
    };
    let (_, orbit) = calculate_orbit(anchor, iterations);

    let list = RefList::new();
    list.push_front(RefOrbit { delta_corr: Complex::ZERO, orbit });
    let gref = GroupRef { list, anchor_px };

    let mut c = cache.lock();
    if c.iterations != iterations {
        c.iterations = iterations;
        c.map.clear();
    }
    if c.map.len() >= GROUP_CACHE_CAP { c.map.clear(); }
    c.map.entry(gkey.clone()).or_insert(gref).clone()
}

/// Render all tiles visible in `coords`, coarse-to-fine, returning early when
/// interrupted. For each progressive pass the entire set of tiles needing that
/// pass is handed to the backend as a single batch. Async like the backends
/// (see `BatchFuture`).
pub async fn run_generation(
    store:       &TileStore,
    group_cache: &Mutex<GroupCache>,
    width:       usize,
    height:      usize,
    coords:      &CoordinatesBox,
    iterations:  usize,
    cursor:      Option<Point<usize, Pixels>>,
    int:         MultiInterrupter<'_>,
    backend:     &dyn Perturbator,
) {
    if store.take_reset() {
        store.clear();
        group_cache.lock().clear();
        info!("reset: dumped all tiles and reference lists");
    }

    let generation = store.begin_generation();
    let view       = coords.view.inner;
    let depth      = store.depth_for_view(view);

    info!(
        "render generation {generation}: depth {depth}, iterations {iterations} \
         [diag: view {view:.4e} = 2^{:.3}, upp 2^{}]",
        view.log2(), crate::tiles::store::upp_log2(depth),
    );

    let x0       = tile_index(&coords.origin.x, depth);
    let y0       = tile_index(&coords.origin.y, depth);
    let tile_px  = TILE_SIZE as f64 * (units_per_pixel(depth) / view);
    let origin00 = TileKey { depth, x: x0.clone(), y: y0.clone() }.origin();
    let sx0      = (&origin00.x - &coords.origin.x).to_f64().value() / view;
    let sy0      = (&origin00.y - &coords.origin.y).to_f64().value() / view;
    let nx       = ((width  as f64 - sx0) / tile_px).ceil().max(1.0) as i64;
    let ny       = ((height as f64 - sy0) / tile_px).ceil().max(1.0) as i64;

    let center = cursor.unwrap_or(Point::new(width / 2, height / 2));

    // Collect and sort tiles by distance from cursor (ascending).
    let mut tiles: Vec<(OrderedFloat<f64>, Arc<crate::tiles::store::Tile>)> = Vec::new();
    let mut seeded = 0usize;
    for (i, j) in iproduct!(0..nx, 0..ny) {
        let key = TileKey { depth, x: &x0 + IBig::from(i), y: &y0 + IBig::from(j) };
        let tile = store.get_or_insert(&key, iterations, generation);
        tile.render_gen.store(generation, std::sync::atomic::Ordering::Relaxed);

        tile.retarget(iterations); // no-op unless the iteration count changed
        if !tile.is_complete() && store.seed_from_relatives(&tile) { seeded += 1; }
        if tile.is_complete() { continue; }

        let cx = sx0 + (i as f64 + 0.5) * tile_px - center.x as f64;
        let cy = sy0 + (j as f64 + 0.5) * tile_px - center.y as f64;
        tiles.push((OrderedFloat(cx * cx + cy * cy), tile));
    }
    tiles.sort_unstable_by_key(|(dist, _)| *dist);

    store.set_view_tiles((nx * ny) as usize);
    store.evict_excess();
    if seeded > 0 { store.bump_progress(); }
    info!("rendering {} tiles ({seeded} seeded from parent/children)", tiles.len());

    let ctx = PassBatchCtx {
        coords:     coords.clone(),
        depth,
        x0:         x0.clone(),
        y0:         y0.clone(),
        width,
        height,
        iterations,
        progress:   store.progress_counter(),
    };

    for pass in 0..NUM_PASSES {
        if int.interrupted() { return; }

        // Build the batch: all tiles that still need this pass, in cursor order.
        let batch: Vec<TileItem> = tiles
            .iter()
            .filter(|(_, tile)| tile.passes_done() <= pass)
            .map(|(_, tile)| {
                let gref = group_list(group_cache, &group_of(&tile.key), iterations);
                TileItem {
                    tile:      tile.clone(),
                    refs:      gref.list,
                    anchor_px: gref.anchor_px,
                }
            })
            .collect();

        if batch.is_empty() { continue; }

        backend.render_pass_batch(&ctx, &batch, pass, &int).await;
        store.bump_progress();
        log_tile_colors(generation, pass, &tiles);
    }

    store.bump_progress();
}

/// DIAG: backend-agnostic check of what actually landed in the visible tiles
/// after a pass: how many pixels are set, how many distinct colours, and the
/// share of the most common one (values are escape iterations). A
/// "monochrome screen" shows up as distinct≈1–2 here if the compute side
/// produced it.
fn log_tile_colors(
    generation: impl std::fmt::Display,
    pass:       u8,
    tiles:      &[(OrderedFloat<f64>, Arc<crate::tiles::store::Tile>)],
) {
    let mut counts = HashMap::<u32, usize>::new();
    let mut unset  = 0usize;
    for (_, tile) in tiles {
        for idx in 0..crate::tiles::store::TILE_LEN {
            match tile.load(idx).get() {
                Some(c) => *counts.entry(c).or_default() += 1,
                None    => unset += 1,
            }
        }
    }
    let set: usize = counts.values().sum();
    let (top, top_n) = counts.iter().max_by_key(|(_, n)| **n).map(|(c, n)| (*c, *n)).unwrap_or((0, 0));
    info!(
        "[diag tiles] gen {generation} pass {pass}: {} tiles, set {set}, unset {unset}, \
         distinct values {}, top value {top} @ {:.1}%",
        tiles.len(), counts.len(), 100.0 * top_n as f64 / set.max(1) as f64,
    );
}
