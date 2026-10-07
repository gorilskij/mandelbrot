# Tiles

The image is computed in square tiles on a quadtree of zoom depths, kept in a shared cache
(`TileStore`), filled pass by pass, and drawn by the [compositor](compositor.md). Code:
`src/tiles/store.rs` (grid, store, passes) and `src/tiles/render.rs` (one generation).

## The grid

| name | value | meaning |
|---|---|---|
| `TILE_SIZE` | 128 (`1 << TILE_POW`, `TILE_POW = 7`) | tile side, in pixels |
| `GROUP_TILES` | 16 (`1 << GROUP_POW`) | tiles per group side: 16 × 16 tiles share one CPU reference list |
| depth d | integer | zoom level: units per pixel `upp(d) = 2^upp_log2(d)`, `upp_log2(d) = −5 − d` |

`pixel_to_coord` gives the exact `FBig` coordinate of a global pixel. `depth_for_view(view)`
picks the depth for a view (units per screen pixel); the store's own `depth_for_view` first
divides by the sampling ratio s ([compositor](compositor.md#sampling-and-antialiasing)).

## Passes

A tile is computed in `NUM_PASSES = 7` passes, coarse to fine, so a usable picture appears
early:

- **Grid passes** with `GRID_STRIDES = [16, 8, 4, 2]`: every 16th pixel, then the new pixels
  of the 8-grid, and so on.
- **The stride-1 step in 3 equal sub-passes**: cell centres first, so every later gap has all
  four axis neighbours (the compositor fills not-yet-computed gaps from them).

`pass_pixels(pass)` is the **single source of truth** for which pixels a pass computes (both
backends use it); `pass_of(r, c)` is its inverse.

**Render order is pass-major**: each pass completes over the visible tiles, rippling out from
the cursor, before the next starts (the owner rejected ring-by-ring refinement, 2026-09-24).

## A generation

`run_generation` (`render.rs`) is one render of a requested view: it lists the visible tiles,
retargets and seeds them, sorts them by distance from the cursor, then for each pass calls
`backend.render_pass_batch(ctx, &batch, pass, &int)` **once** with every tile that still
needs that pass. The
compute thread runs one generation per requested view and abandons it when a newer view
arrives (the interrupt is checked between chunks of work).

## Iterations, not colours

Tiles store **escape iterations**: `0` = in the set, `n` = escaped at iteration n, bit 31 =
not computed yet (`MaybePixel`, `src/drawing/maybe_pixel.rs`). Colour is applied later
([compositor](compositor.md#colouring-on-the-gpu)), so the palette can change for free.

Changing the maximum iterations calls `Tile::retarget`:

- **↓ (fewer)** is exact with no recompute: any `n > max` becomes "in the set".
- **↑ (more)** clears only the in-set pixels, lowers `passes_done` to the prefix of passes
  still complete, and keeps displaying at the old level (`display_floor`; cleared pixels draw
  black until recomputed).

`Tile::version` changes whenever a tile's displayable contents change (the compositor
re-uploads on it).

## Quadtree seeding

Pixel (i, j) at depth d is pixel (2i, 2j) at depth d + 1. So a finished parent gives a child
passes 0–3 for free, and four finished children give their parent
(`seed_from_relatives`). Only between tiles computed with the same iteration count.

## The memory budget

The store evicts least-recently-drawn tiles (`evict_excess`, at the start of each
generation; never the current generation's tiles) down to a budget:

- **max(1 GiB, 3 × the current view's tiles)** (`MEMORY_BUDGET_BYTES`, the 3× cache).
- **On the web only**, capped by the **memory ceiling** of 3 GiB (`memory_ceiling_bytes`),
  counting each tile's CPU pixels, GPU iterations and colour levels (`TILE_TOTAL_BYTES`).
  wasm32 has 4 GiB in all. **Natively there is no ceiling** (owner: "the full ungodly glory of
  s = 4").
- The view's own tiles cannot be evicted, so on the web the ceiling also bounds s: ←/→ set the
  **requested** s; the store uses the largest 0.5 step at or below it whose worst-case view
  fits (`effective_ratio`, recomputed on resize and s changes; shrinking the window restores
  it). The 3× cache shrinks first. **Never silent** (owner's rule): the title shows `s 4 (→3)`
  and `cache 1.4×` when cut, and the log warns.

For scale: s = 4 needs up to ~24k tiles for a 3000 × 2000 screen, ~15 GiB in all copies with
the 3× cache (the GPU keeps each tile's iterations too). With a fixed 1 GiB budget the view
alone filled it, evicting parents and recent views (black previews while moving, no reuse) —
hence the budget that follows the view.
