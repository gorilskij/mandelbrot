//! CPU perturbation backend: f32 delta iteration with on-the-fly promotion of
//! new reference orbits (the original algorithm). A glitched pixel computes
//! its own high-precision orbit, projects it, and pushes it into the shared
//! group list so later pixels reuse it.

use super::{PassBatchCtx, Perturbator, RefList, RefOrbit, TileItem};
use crate::rendering::{Pf, calculate_orbit, check_divergence_delta, check_orbit};
use crate::tiles::store::{
    GROUP_POW, GROUP_TILES, NUM_PASSES, TILE_SIZE, Tile, floor_div_pow2, pass_pixels,
    pixel_to_coord, units_per_pixel, working_precision,
};
use dashu::integer::IBig;
use itertools::iproduct;
use num::Complex;
use rayon::prelude::*;
use std::num::NonZeroUsize;
use waker_interrupter::MultiInterrupter;

/// When true, render tiles sequentially in a fixed order (reproducible output).
const DETERMINISTIC: bool = false;

/// CPU perturbation backend.
pub struct Cpu;

impl Perturbator for Cpu {
    fn render_pass_batch(
        &self,
        _ctx:  &PassBatchCtx,
        tiles: &[TileItem],
        pass:  u8,
        int:   &MultiInterrupter,
    ) {
        if DETERMINISTIC {
            for item in tiles {
                if int.interrupted() { return; }
                render_tile_pass(&item.tile, &item.refs, item.anchor_px, pass, _ctx.iterations, int);
            }
        } else {
            tiles.par_iter().for_each(|item| {
                if !int.interrupted() {
                    render_tile_pass(
                        &item.tile, &item.refs, item.anchor_px,
                        pass, _ctx.iterations, int,
                    );
                }
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Per-tile implementation (unchanged from original)
// ---------------------------------------------------------------------------

/// Everything fixed about a tile for the duration of a render pass.
struct TileCtx<'a> {
    tile: &'a Tile,
    refs: &'a RefList,
    /// units per tile pixel
    upp: f64,
    /// pixel offset of this tile's (0,0) pixel from the group anchor
    anchor_dx: i64,
    anchor_dy: i64,
    /// global pixel coordinate of this tile's (0,0) pixel
    tile_px0_x: IBig,
    tile_px0_y: IBig,
    depth: i64,
    prec:  usize,
}

/// Render one pixel (or skip if already computed). Returns true if black.
fn render_pixel(ctx: &TileCtx, c: usize, r: usize, iterations: usize) -> bool {
    let idx = r * TILE_SIZE + c;

    if let Some(raw) = ctx.tile.load(idx).get() {
        return raw == 0;
    }

    let delta = Complex {
        re: (ctx.anchor_dx + c as i64) as Pf * ctx.upp as Pf,
        im: (ctx.anchor_dy + r as i64) as Pf * ctx.upp as Pf,
    };

    let val: Option<NonZeroUsize> = ctx
        .refs
        .iter()
        .find_map(|ref_orbit| {
            check_divergence_delta(&ref_orbit.orbit, delta - ref_orbit.delta_corr).ok()
        })
        .unwrap_or_else(|| {
            let x_0 = Complex {
                re: pixel_to_coord(&ctx.tile_px0_x + IBig::from(c as u64), ctx.depth, ctx.prec),
                im: pixel_to_coord(&ctx.tile_px0_y + IBig::from(r as u64), ctx.depth, ctx.prec),
            };
            let (_, new_orbit) = calculate_orbit(x_0, iterations);
            let val = check_orbit(&new_orbit).unwrap();
            ctx.refs.push_front(RefOrbit { delta_corr: delta, orbit: new_orbit });
            val
        });

    ctx.tile.store(idx, (val.map_or(0, |n| n.get()) as u32).into());
    val.is_none()
}

/// Run one progressive pass over a tile.
fn render_tile_pass(
    tile:      &Tile,
    refs:      &RefList,
    anchor_px: (i64, i64),
    pass:      u8,
    iterations: usize,
    int:       &MultiInterrupter,
) -> bool {
    let gx = floor_div_pow2(&tile.key.x, GROUP_POW);
    let gy = floor_div_pow2(&tile.key.y, GROUP_POW);
    let local_x = i64::try_from(&(&tile.key.x - &gx * IBig::from(GROUP_TILES as u64)))
        .expect("tile-in-group offset fits i64");
    let local_y = i64::try_from(&(&tile.key.y - &gy * IBig::from(GROUP_TILES as u64)))
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
    let mut last_row = usize::MAX;
    for (r, c) in pass_pixels(pass) {
        if r != last_row {
            if int.interrupted() { return false; }
            last_row = r;
        }
        all_black &= render_pixel(&ctx, c, r, iterations);
    }

    // Black-fill: if all coarse samples are black, check the perimeter;
    // if that is also black, fill the whole tile and skip remaining passes.
    if pass == 0 && all_black {
        let mut perimeter_black = true;
        for r in 0..TILE_SIZE {
            if int.interrupted() { return false; }
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
