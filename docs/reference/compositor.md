# The compositor

`src/gpu_compositor.rs` (with `gpu_compositor_recolour.wgsl` and
`gpu_compositor_downsample.wgsl`) draws the tiles to the screen with **its own wgpu device**,
separate from the compute backend's. Natively it runs on a render thread; on the web on the
browser's main thread, on each animation frame (`WebRenderer`; [web build](web-build.md)).

## Colouring on the GPU

- When a tile's `version` changes, its iteration counts are uploaded (`iteration_texels`: the
  stride grid, or the whole tile; `0` = black, including pixels being recomputed after ↑;
  `NONE_RAW` = a sub-pass gap not computed yet).
- A compute pass (`gpu_compositor_recolour.wgsl`) makes the tile's colour texture levels:
  palette, **gap fill** (a gap gets the mean of its computed axis neighbours) and mip averaging,
  all in **linear light**.
- **A palette change** (`palette_gen`) re-runs only that pass: nothing is recomputed or
  re-uploaded.
- **The palette is computed in the shader** (`colour`, from a 16-byte uniform of the two
  phases). `rendering::val_to_color` is the test-only CPU reference; `recolour_matches_cpu`
  keeps them in sync — **change both**. (A palette lookup table was the pass's main cost on
  noisy tiles: a gather by iteration.)
- **RAM:** the iterations stay on the GPU (64 KiB per full tile; ~1.5 GiB at s = 4), accepted by
  the owner for fast recolouring.

## Chunks

Tiles of one `Shape` (data size, drawn levels) share **2D array textures** (`Chunk`, between
`CHUNK_MIN_LAYERS` = 16 and `CHUNK_MAX_LAYERS` = 256 layers, a slot per tile, dropped once
empty). Colouring is one dispatch per chunk and level (layer lists via a dynamic offset);
drawing is one draw call per chunk. Why: Metal serialises dispatches (~10–15 µs each), and
per-tile dispatches took 365 ms for a recolour at s = 4. Measured with `diag_recolour_speed`
(M2 Pro, full 3000 × 2000 screen): s = 4 (24 000 tiles) 25–30 ms, s = 1 ~10 ms.

A chunk is dropped only when entirely empty, so eviction can leave sparse ones
([backlog](../backlog.md)).

## Sampling and antialiasing

- **The sampling ratio s** (0.5…4 in steps of 0.5, default 1; ←/→): tiles are rendered at the
  depth where a screen pixel spans r ∈ [s, 2s) tile pixels per axis (`TileStore::min_ratio`,
  `depth_for_view`; s is shared with the compute thread through the store). s = 0.5 computes
  at most 1:1 (~¼–1× the screen's pixels); s = 1 computes 1–4×; s = 4 up to 16–64×.
- Tile colour textures are sampled as **sRGB**; each level is the area average, in linear
  light, of the data texels it covers. A tile holds only the one or two levels drawn at the
  current s (`drawn_levels`: level 0 at s ≤ 1, level 2 at s = 4 — 1/16 of the texels; finer
  ones as needed for parent previews). Changing s changes the tiles' shapes, so they are
  re-uploaded and recoloured.
- Tiles are drawn **1:1 into an offscreen sRGB image** (≤ 2× the surface per axis) from the
  level k that leaves q = r/2^k ∈ [1, 2) texels per screen pixel, grid-aligned. The
  **downsample pass** (`gpu_compositor_downsample.wgsl`) takes each screen pixel's exact q × q
  area average (bilinear below q = 1) and encodes to sRGB (the surface is not sRGB).
- Cost vs plain bilinear (start-up view, 3000 × 2000): ~+2 ms of compositor GPU time per frame
  (median 2.9 → 4.8 ms); compute ~3–5 % slower from sharing the GPU.

## The progress bar

A white bar filling in proportion to the pixel work done (`BarAnim`). Rising, it moves at the
measured progress speed, so updates seconds apart do not show as jumps, catching up over
about one update interval when far behind; it has momentum and trails the real progress by
~2 update intervals so irregular updates do not make it jerk; it is never ahead of the real
progress. Falling (a restart), it moves as a critically damped spring. `render()` returns
whether the bar is still moving, and the render thread keeps drawing at vsync until it settles.
