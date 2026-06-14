//! Tile rendering orchestration: enumerate the tiles intersecting the
//! viewport, manage the per-group reference lists, and drive the progressive
//! passes. The per-pixel perturbation evaluation is delegated to a
//! `Perturbator` backend (CPU / GPU sister modules under `perturb`).
//!
//! Tiles are bundled into large square *groups* (GROUP_TILES x GROUP_TILES
//! tiles). All tiles in a group share one reference-orbit list — the original
//! whole-screen concurrent (lock-free append-only) list, keyed per group. The
//! group's seed reference is the FBig orbit of a point in the group (its
//! center normally; a RANDOM point in TEST/spacebar mode).

use crate::rendering::{CoordinatesBox, Pixels, calculate_orbit};
use crate::support::Point;
use crate::tiles::perturb::{Perturbator, RefList, RefOrbit};
use crate::tiles::store::{
    GROUP_POW, GROUP_TILES, NUM_PASSES, TILE_SIZE, Tile, TileKey, TileStore, depth_for_view,
    floor_div_pow2, pixel_to_coord, tile_index, units_per_pixel, working_precision,
};
use dashu::integer::IBig;
use itertools::iproduct;
use log::info;
use num::Complex;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use rayon::ThreadPool;
use std::collections::HashMap;
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

/// A group's shared reference list plus its anchor (the pixel offset, within
/// the group, that deltas are measured from — the seed reference's location).
#[derive(Clone)]
struct GroupRef {
    list: RefList,
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
        let pick =
            || ((rand::random::<f64>() * GROUP_SIDE_PX as f64) as i64).clamp(0, GROUP_SIDE_PX - 1);
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

    let list = RefList::new();
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
    backend: &(dyn Perturbator + Send),
) {
    // TEST: spacebar requests a full reset — dump every tile and every group
    // reference list, so everything is recomputed with fresh references
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
                if backend.render_tile_pass(tile, &gref.list, gref.anchor_px, pass, iterations, &int)
                {
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
                        if backend.render_tile_pass(
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
