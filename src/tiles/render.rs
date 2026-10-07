//! Tile rendering orchestration: enumerate the tiles intersecting the
//! viewport and drive the progressive passes. The actual per-pixel work is delegated to a `Perturbator` backend
//! via `render_pass_batch` — one call per pass with *all* tiles that need it.

use crate::rendering::{CoordinatesBox, Pixels};
use crate::support::Point;
use crate::tiles::perturb::{PassBatchCtx, Perturbator, TileItem};
use crate::tiles::store::{NUM_PASSES, TILE_SIZE, TileKey, TileStore, tile_index, units_per_pixel};
use dashu::integer::IBig;
use itertools::iproduct;
use log::info;
use ordered_float::OrderedFloat;
use std::sync::Arc;
use waker_interrupter::MultiInterrupter;

/// Render all tiles visible in `coords`, coarse-to-fine, returning early when
/// interrupted. For each progressive pass the entire set of tiles needing that
/// pass is handed to the backend as a single batch. Async like the backends
/// (see `BatchFuture`).
pub async fn run_generation(
    store:       &TileStore,
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
        backend.reset();
        info!("reset: dumped all tiles, reference lists and the GPU's cached reference");
    }

    let generation = store.begin_generation();
    let view       = coords.view.inner;
    let depth      = store.depth_for_view(view);
    let t_gen      = web_time::Instant::now(); // DIAG
    let t_first    = *FIRST_GENERATION.get_or_init(|| t_gen); // DIAG

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
            .map(|(_, tile)| TileItem { tile: tile.clone() })
            .collect();

        if batch.is_empty() { continue; }

        backend.render_pass_batch(&ctx, &batch, pass, &int).await;
        store.bump_progress();
        info!( // DIAG
            "[diag tiles] gen {generation} pass {pass} done: {:.0} ms into the generation, {:.0} ms since the first",
            t_gen.elapsed().as_secs_f64() * 1e3, t_first.elapsed().as_secs_f64() * 1e3,
        );
    }

    store.bump_progress();
}

/// DIAG: when the first generation started, for timing a whole render
/// across the generations that restart it.
static FIRST_GENERATION: std::sync::OnceLock<web_time::Instant> = std::sync::OnceLock::new();
