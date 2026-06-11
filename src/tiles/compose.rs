//! Compositor: assembles the display buffer for an arbitrary viewport from
//! whatever tiles are available. For every visible tile position it uses, in
//! order of preference:
//!   1. the tile itself (at whatever progressive stride it has completed),
//!   2. an ancestor tile, upscaled (the normal preview while zooming in),
//!   3. descendant tiles, downscaled (the normal preview while zooming out),
//!   4. black.
//! Sampling is bilinear over the tile's completed stride grid, so a tile that
//! has only finished its 16-pixel pass shows up as a smooth, low-resolution
//! image that sharpens as finer passes complete.

use crate::rendering::{CoordinatesBox, interpolate};
use crate::tiles::store::{
    TILE_SIZE, Tile, TileKey, TileStore, depth_for_view, tile_index, units_per_pixel,
};
use dashu::integer::IBig;
use itertools::iproduct;
use std::sync::Arc;

/// how many levels up to look for a coarser fallback tile
const MAX_CLIMB: usize = 40;
/// how many levels down to look for finer fallback tiles (4^k lookups!)
const MAX_DESCEND: usize = 3;

/// integer pixel range [start, end) covered by the half-open screen-space
/// interval [a, b), clamped to [0, limit)
fn pixel_range(a: f64, b: f64, limit: usize) -> (usize, usize) {
    let start = (a.ceil().max(0.0) as usize).min(limit);
    let end = (b.ceil().max(0.0) as usize).min(limit);
    (start, end)
}

pub fn compose(
    store: &TileStore,
    coords: &CoordinatesBox,
    width: usize,
    height: usize,
    out: &mut [u32],
) {
    let frame = store.next_frame();
    let view = coords.view.inner;
    let depth = depth_for_view(view);

    // screen-space size of one tile, in (64, 128]
    let tile_px = TILE_SIZE as f64 * (units_per_pixel(depth) / view);

    // top-left visible tile and its screen position
    let x0 = tile_index(&coords.origin.x, depth);
    let y0 = tile_index(&coords.origin.y, depth);
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

    for (i, j) in iproduct!(0..nx, 0..ny) {
        let key = TileKey {
            depth,
            x: &x0 + IBig::from(i),
            y: &y0 + IBig::from(j),
        };
        let rect = Rect {
            x: sx0 + i as f64 * tile_px,
            y: sy0 + j as f64 * tile_px,
            side: tile_px,
        };
        compose_tile(store, frame, &key, rect, width, height, out, MAX_CLIMB, MAX_DESCEND);
    }
}

/// screen-space square covered by a tile
#[derive(Copy, Clone)]
struct Rect {
    x: f64,
    y: f64,
    side: f64,
}

fn compose_tile(
    store: &TileStore,
    frame: u64,
    key: &TileKey,
    rect: Rect,
    width: usize,
    height: usize,
    out: &mut [u32],
    climb_budget: usize,
    descend_budget: usize,
) {
    let (px_start, px_end) = pixel_range(rect.x, rect.x + rect.side, width);
    let (py_start, py_end) = pixel_range(rect.y, rect.y + rect.side, height);
    if px_start >= px_end || py_start >= py_end {
        return;
    }

    // 1-2. the tile itself or an ancestor. Among everything available up the
    // chain, pick the source with the finest *actual* sample spacing rather
    // than blindly preferring the target tile: a freshly-started target tile
    // that has only finished its coarse pass is lower resolution than a fully
    // rendered parent upscaled 2x, and switching to it would make the image
    // visibly "pop" (jump) at every depth boundary while zooming. The
    // effective stride, measured in target-tile pixels, is the tile's
    // completed stride times 2^(levels climbed); the smaller, the sharper.
    // Ties favor the deeper (native) tile so we don't cling to ancestors.
    let mut candidate = key.clone();
    let mut candidate_rect = rect;
    let mut best: Option<(usize, Arc<Tile>, Rect)> = None;
    let mut best_eff = f64::INFINITY;
    for climb in 0..=climb_budget {
        if let Some(tile) = store.get(&candidate) {
            if let Some(stride) = tile.completed_stride() {
                let eff = stride as f64 * (1u64 << climb.min(52)) as f64;
                if eff < best_eff {
                    best_eff = eff;
                    best = Some((stride, tile, candidate_rect));
                }
            }
        }
        // the finest possible source at the next level up has effective
        // stride 2^(climb+1); if our best already matches that, stop climbing
        if best_eff <= (1u64 << (climb + 1).min(52)) as f64 {
            break;
        }

        // grow the rect to the parent tile's screen-space square
        let (bx, by) = candidate.parent_offset();
        candidate_rect = Rect {
            x: candidate_rect.x - bx as f64 * candidate_rect.side,
            y: candidate_rect.y - by as f64 * candidate_rect.side,
            side: candidate_rect.side * 2.0,
        };
        candidate = candidate.parent();
    }

    if let Some((stride, tile, src_rect)) = best {
        tile.touch(frame);
        sample_tile_rect(
            &tile,
            stride,
            src_rect,
            (px_start, px_end),
            (py_start, py_end),
            width,
            out,
        );
        return;
    }

    // 3. descendants (the preview available after zooming out)
    if descend_budget > 0 {
        let half = rect.side / 2.0;
        for (cx, cy) in iproduct!(0..2u64, 0..2u64) {
            let child_rect = Rect {
                x: rect.x + cx as f64 * half,
                y: rect.y + cy as f64 * half,
                side: half,
            };
            // no climb: we already know this tile's ancestors have no data
            compose_tile(
                store,
                frame,
                &key.child(cx, cy),
                child_rect,
                width,
                height,
                out,
                0,
                descend_budget - 1,
            );
        }
        return;
    }

    // 4. nothing available
    for py in py_start..py_end {
        out[py * width + px_start..py * width + px_end].fill(0);
    }
}

/// Fill `out[px_range x py_range]` by bilinearly sampling `tile`, whose
/// screen-space square is `rect`, over its completed grid of every
/// `stride`-th pixel.
fn sample_tile_rect(
    tile: &Arc<Tile>,
    stride: usize,
    rect: Rect,
    (px_start, px_end): (usize, usize),
    (py_start, py_end): (usize, usize),
    width: usize,
    out: &mut [u32],
) {
    let scale = TILE_SIZE as f64 / rect.side;
    let s = stride as f64;
    // last grid point of the completed stride grid
    let gmax = TILE_SIZE - stride;

    // grid coordinates and interpolation weight along one axis
    let split = |screen: usize, rect_start: f64| -> (usize, usize, f64) {
        let a = ((screen as f64 - rect_start) * scale).clamp(0.0, (TILE_SIZE - 1) as f64);
        let g0 = (((a / s).floor() * s) as usize).min(gmax);
        let g1 = (g0 + stride).min(gmax);
        let frac = if g1 > g0 {
            ((a - g0 as f64) / (g1 - g0) as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (g0, g1, frac)
    };

    for py in py_start..py_end {
        let (gy0, gy1, fy) = split(py, rect.y);
        for px in px_start..px_end {
            let (gx0, gx1, fx) = split(px, rect.x);

            let color = interpolate(
                tile.load(gy0 * TILE_SIZE + gx0), // top-left
                tile.load(gy0 * TILE_SIZE + gx1), // top-right
                tile.load(gy1 * TILE_SIZE + gx0), // bottom-left
                tile.load(gy1 * TILE_SIZE + gx1), // bottom-right
                (fy, fx),
            );

            out[py * width + px] = color.get().unwrap_or(0);
        }
    }
}
