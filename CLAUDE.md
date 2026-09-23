# Mandelbrot explorer

Interactive deep-zoom Mandelbrot viewer in Rust (winit + wgpu). Uses perturbation
theory: one high-precision reference orbit (`dashu` `FBig`), cheap low-precision
delta iteration per pixel. There are two interchangeable backends, CPU and GPU.

## Running

- `cargo run` — `RUST_LOG=info cargo run` to see render-generation logs.
- Controls: **Escape** switches CPU/GPU backend (and resets). **Space** resets,
  which drops every cached tile and reference list. **↑/↓** doubles/halves the
  max iterations (default 2048). **Ctrl/Cmd-C / V** copies/pastes `coords/iterations`.

## Layout

- `src/main.rs` — app, input, view state. Builds `Toggle` (CPU/GPU) + `GpuState`.
- `src/rendering.rs` — `calculate_orbit` (returns `(Orbit<FBig>, Orbit<Pf>)`),
  `check_orbit`, `check_divergence_delta` (the reference CPU delta loop),
  `val_to_color`, and `Pf` (the perturbation working float).
- `src/tiles/store.rs` — tile grid math and `TileStore`. `TILE_SIZE=128`,
  groups of `GROUP_TILES=16`² tiles,
  `units_per_pixel(depth) = 2^upp_log2(depth)`, `upp_log2 = -5 - depth`,
  `pixel_to_coord` (exact FBig coordinate of a global pixel).
  - **Passes** (`NUM_PASSES=7`): grid passes `GRID_STRIDES=[16,8,4,2]`, then
    the stride-1 step split into 3 equal sub-passes (cell centres first, so
    every later gap has all 4 axis neighbours). `pass_pixels(pass)` is the
    single source of truth; both backends use it. The compositor shows grid
    passes via the sampler and, from the first sub-pass, a full-res texture
    with gaps averaged from neighbours (`reconstruct`).
  - **Tiles store escape iterations, not colours** (0 = in set, n = escaped
    at n; bit 31 = not computed). The compositor colours at upload through a
    palette table (`color_of`). Changing iterations calls `Tile::retarget`:
    ↓ is exact with no recompute (n > max → in set); ↑ clears only in-set
    pixels, lowers `passes_done` to the still-complete prefix and keeps
    displaying at the old level (`display_floor`, cleared pixels drawn black).
    The compositor re-uploads on `Tile::version`, not `passes_done`.
  - **Quadtree seeding** (`seed_from_relatives`): pixel (i,j) at depth d is
    pixel (2i,2j) at depth d+1, so a finished parent gives a child passes
    0–3 for free, and four children give their parent. Same iterations only.
- `src/tiles/render.rs` — `run_generation`: lists visible tiles, sorts them by
  distance from the cursor, and for each pass calls
  `backend.render_pass_batch(ctx, &batch, pass, &int)` once with every tile that
  still needs that pass. Owns `GroupCache` (the per-group reference orbits).
- `src/tiles/perturb/mod.rs` — the `Perturbator` trait (`render_pass_batch`),
  `PassBatchCtx`, `TileItem { tile, refs, anchor_px }`, `Toggle`.
- `src/tiles/perturb/cpu.rs` — CPU backend. Runs tiles through `rayon` `par_iter`
  (`DETERMINISTIC` const = sequential). A glitched pixel computes its own exact
  orbit and pushes it onto the group `RefList`. Also does black-fill: if pass 0
  plus the tile perimeter are all black, the whole tile is filled black.
- `src/tiles/perturb/gpu.rs` — GPU backend (details below).
- `src/tiles/perturb/shaders/*.wgsl` — WGSL sources, pulled in with `include_str!`.
- `src/drawing/renderer/multithreaded.rs` — the render thread, which calls `run_generation`.
- `src/gpu_compositor.rs` — draws the tiles to screen. It is separate from the compute backend.

## Perturbation math (both backends)

Take reference `C` with orbit `X₀=C, X_{n+1}=Xₙ²+C`. A pixel sits at `c=C+δ₀`, and
`zₙ = Xₙ + δₙ` with `δ_{n+1} = 2·Xₙ·δₙ + δₙ² + δ₀`. It escapes when `|Xₙ+δₙ|² > 4`.
This convention uses `z₀ = c`, not 0.

**Perturbation is algebraically exact for any size of δ.** A large δ only risks
float precision, so there is **no** `|δ|/|X|` (Pauldelbrot-style) glitch
heuristic. A pixel is "glitched" only when the reference orbit ended early
(`!is_full`) and the pixel has not escaped yet. This matches
`check_divergence_delta`. An earlier `|δ|² > 1e-6·|x|²` check in the shader
flagged nearly every pixel at low zoom and produced an "iteration-1 egg". Do not
re-add it.

**Rebasing** (Zhuoran 2021) is in both GPU shaders: when |z| < |δ|, set
δ ← z² + δ₀ and restart the reference index at 0 (the X₀ = C form of "δ ← z,
back to the start"). It is exact, not a heuristic, and removes the blocky
precision-loss artifacts that appear once the reference is a nucleus the
pixel's orbit does not follow. The shaders track `n` (iterations, reported)
and `m` (reference index) separately. `shaders_validate` checks the WGSL with
naga, since it is otherwise only compiled at runtime.

**Interior detection** (GPU, nucleus references only): the shaders track
log2|dz/dz₀|² and, every window of ≥128 iterations (a whole number of the
nucleus period p), compare it with the previous window; 2 consecutive windows
contracting by ≥0.9× per period → in the set. Inside a component the cycle
multiplier |λ| < 1; just outside |λ| ≥ 1. Short windows (< ~64) produce false
positives (escaping pixels contracting briefly near 0), so don't shorten them.
`interior_detection_is_safe_and_useful` (regular) and `diag_interior_detection`
(ignored, broader) check 0 false positives against exact orbits. Parameters
are the `INTERIOR_*` consts in gpu.rs, passed via uniforms.

## GPU backend (`gpu.rs`)

- For each pass, all needed pixels from all batch tiles are collected in
  cursor order and dispatched in **chunks of ~`TARGET_CHUNK_MS` (30 ms)**,
  sized from the last chunk's measured ms/pixel (floor `MIN_CHUNK_PX`=64k,
  since each dispatch has ~0.8 ms fixed overhead). The interrupt is checked
  between chunks; in-flight GPU work cannot be cancelled. Per chunk: dispatch
  → glitch rounds (up to `MAX_GLITCH_PASSES=8`) → residual CPU resolve →
  store → finish every tile whose pixels are all stored → bump
  `ctx.progress` so the compositor redraws.
- **Reference = a nearby nucleus** (`nucleus.rs`: ball-period detection +
  Newton in FBig). Its orbit never escapes, so normally nothing glitches.
  Seeded from the view centre; if the centre escapes too early to reveal the
  period, glitch rounds evaluate 32 sampled glitched pixels exactly (in
  parallel), take the longest-lived, and seed the search from it. The best
  reference is cached in `GpuState` across passes and nearby views
  (`MAX_REF_DIST` view radii); failed centre searches are remembered per view.
- References need not lie on the pixel grid: `ref_px` gives the reference's
  position in pixel units, computed exactly in FBig; pixel offsets are
  `col - ref_px` for both pipelines.
- Output encoding: `0` = in set, `n` = escaped at iteration n,
  `GLITCH_BIT (0x8000_0000) | n` = glitched.
- **2D dispatch**: `gx=min(groups,65535)`, `gy=ceil(groups/65535)`. The shader
  rebuilds `idx = gid.y*dispatch_w + gid.x`, because a dimension may not exceed 65535.
- **Binding-size chunking**: `dispatch()` splits batches so the delta buffer stays
  under `max_storage_buffer_binding_size` (128 MiB here). It takes an `elem`
  size: 8 B for f32 deltas, 16 B for floatexp seeds.
- **Never cache a guess**: a pixel that still carries `GLITCH_BIT` is *not
  stored*, so the next generation retries it. Leftover glitches are resolved
  exactly on the CPU (`calculate_orbit` + `check_orbit`), in parallel and
  interruptibly. Storing glitches as black and then skipping them
  (`is_some()`) across interrupted generations caused wrong-iteration pixels
  mixed into correct ones while zooming.
- A tile's `finish_pass` runs as soon as every pixel it needed this pass is
  stored (even mid-pass or when interrupted), and only if it has finished the
  previous pass. A tile with an unstored glitch never finishes, so it is
  retried. The compositor only re-uploads a tile when `passes_done` changes.
- The black-fill optimisation is intentionally absent on the GPU.

### Precision / floatexp

- f32 hits a hard wall at the **exponent**, not the mantissa. Once
  units-per-pixel falls below ~2⁻¹²⁶ (≈1e-38) the seed δ₀ is subnormal, and
  neighbouring pixels collapse to one value. This was observed at a zoom of ~7.5e-41.
  Double-single (hi+lo f32) would **not** help, because it keeps the f32 exponent range.
- Fix: shared-exponent complex **floatexp** (`Fe { m: vec2<f32>, e: i32 }`, in
  `shaders/floatexp.wgsl`). The deep shader `perturbation_floatexp.wgsl` works in
  two phases: iterate δ in floatexp until `e > -100`, then continue in plain f32
  (δ₀ is negligible by that point).
- The pipeline is chosen per pass: `use_fe = upp_log2(depth) < FE_THRESHOLD (-100)`.
  The shallow f32 pipeline is left untouched.
- `pack_fe(dx, dy, base_exp)`: seeds are built in **pixel units** (O(1e4), so the
  f64 math is always safe), and the depth scale `2^upp` goes into the exponent.
  Never form `2^upp` as an f64 on the deep path, because it underflows.
- The reference orbit stays plain f32. Its values are O(1).
- **Mantissa precision is the remaining accuracy limit** (measured by the
  ignored test `diag_reference_precision`): against a full nucleus reference,
  ~30% of long-lived near-boundary pixels get a different escape count in f32
  than exact (1/256 in f64). Mostly from rounding the reference orbit to f32,
  but a hi+lo f32 reference alone does not fix it; the delta arithmetic needs
  more precision too. Pauldelbrot's |z|≪|X| test does not catch these.

### Conventions

- WGSL lives in `.wgsl` files and is loaded with `include_str!`. Shared helpers
  are prepended with `format!("{FE_HELPERS}\n{BODY}")`. Keep shader code out of
  Rust string literals.
- `Pf` in `rendering.rs` is currently **`f32`**. That was set on purpose so the
  CPU matches GPU precision for comparison testing. The CPU's normal setting is
  `f64`, which reaches far deeper. Ask before changing it.

## Open threads (as of 2026-09-23)

Talk these through before coding:

1. **Done** on branch `fix-glitches`: bounded, interruptible chunked dispatch
   with per-chunk progress (was: whole-screen dispatch blocked on
   `poll(wait_indefinitely)` for up to ~1.7 s).
2. **Ring-by-ring refinement** (not done): chunks are currently pass-major
   (all of pass N, cursor-first, then pass N+1). The preferred order is pass 0
   everywhere first, then refine ring by ring around the cursor, finishing each
   ring including its glitch correction. A nested loop should be enough.
3. **Cached group references on the GPU**: largely superseded by the nucleus
   reference (no glitches, no per-pass FBig orbits). The GPU still ignores
   `TileItem.refs`.
4. **Mantissa precision** (see Precision above): double-single arithmetic for
   the delta loop, or accept f32 accuracy. Undecided.

Other known gaps: the GPU has no black-fill.
