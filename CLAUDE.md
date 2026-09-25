# Mandelbrot explorer

Interactive deep-zoom Mandelbrot viewer in Rust (winit + wgpu). Uses perturbation
theory: one high-precision reference orbit (`dashu` `FBig`), cheap low-precision
delta iteration per pixel. There are two interchangeable backends, CPU and GPU;
the GPU one is the main one (nucleus references, rebasing, BLA, interior
detection); the CPU one is older (per-group references, no BLA).

## Running

- `cargo run --release` (dev builds optimise dependencies too, see Cargo.toml).
  Logging defaults to `info` and is teed to `gpu.log` in the working directory
  (overwritten on each launch, gitignored); `RUST_LOG` overrides the level.
- The app **starts on the GPU backend**; **Escape** switches CPU/GPU (and
  resets). **Space** resets: drops every tile and the CPU reference lists (the
  GPU backend's cached reference survives, which is harmless). **↑/↓**
  doubles/halves the max iterations (default 2048; tiles are retargeted, not
  recomputed, see below). **←/→** change the requested sampling ratio s by 0.5
  (0.5…4, default 1; see the compositor), lowered if the view would not
  fit the memory ceiling (wasm only; see the tile budget; title `s 4 (→3)`). **Cmd-scroll** shifts
  the phase of the palette's hue sine (1/128 turn per notch), **Ctrl-scroll**
  the lightness one (1/32; Alt is bound elsewhere on the user's machine)
  (`PalettePhase`; recolours only; both together shift both). **Cmd/Ctrl-C / V** copies/pastes `coords/iterations`
  (`x,y|units_per_pixel/iterations`, `x,y` = top-left corner). Letter shortcuts
  match the layout's character (`logical_key`), not the key position (the user
  types Dvorak).
- Diagnostics (marked `DIAG` in the code): the window title shows backend /
  pipeline (`cpu`, `f32`, `fe`), view, depth, upp, iterations, centre (f64,
  too coarse to navigate back: use Cmd-C), held modifiers and the last key with
  what it did. `[diag gpu]` / `[diag tiles]` log lines give per-pass and
  per-chunk timings, references, glitch rounds and result statistics.

## Layout

- `src/main.rs` — app, input, view state, title/log diagnostics.
- `src/platform.rs` — everything that differs between native and the web
  build (see Web build): spawning threads/workers, sleeping, logging,
  clipboard, the debug line, UI-thread maps.
- `web/` — the web build: `build.sh`, `serve.py`, `index.html`, `_headers`,
  README.
- `src/rendering.rs` — `calculate_orbit` (returns `(Orbit<FBig>, Orbit<Pf>)`),
  `check_orbit`, `check_divergence_delta` (the reference CPU delta loop),
  `val_to_color` (the palette), and `Pf` (the perturbation working float).
- `src/tiles/store.rs` — tile grid math and `TileStore`. `TILE_SIZE=128`,
  groups of `GROUP_TILES=16`² tiles,
  `units_per_pixel(depth) = 2^upp_log2(depth)`, `upp_log2 = -5 - depth`,
  `pixel_to_coord` (exact FBig coordinate of a global pixel).
  - **Tile budget** (LRU eviction at the start of each generation, never the
    current generation's tiles): max(1 GiB, 3× the current view's tiles),
    capped by the **memory ceiling** on wasm only (`memory_ceiling_bytes`:
    3 GiB, counting each tile's CPU pixels, GPU iterations and colour
    levels, `TILE_TOTAL_BYTES`). **Natively there is no ceiling** (user:
    "the full ungodly glory of s = 4"). The view's own tiles
    can't be evicted, so the ceiling also bounds s: ←/→ set the requested
    s, the store uses the largest 0.5 step at or below it whose worst-case
    view fits (`effective_ratio`, on resize and s changes; shrinking the
    window restores it). The 3× cache shrinks first. Never silent (user):
    title `s 4 (→3)`, `cache 1.4×` when cut, a warning in the log. A high
    s needs many tiles (s = 4: up to ~24k for 3000×2000, ~15 GiB in all
    copies with the 3× cache, since the GPU keeps each tile's iterations
    too); with a fixed 1 GiB the
    view alone filled it, evicting parents and recent views (black previews
    while moving, no reuse). Lowering s shrinks it again.
  - **Passes** (`NUM_PASSES=7`): grid passes `GRID_STRIDES=[16,8,4,2]`, then
    the stride-1 step split into 3 equal sub-passes (cell centres first, so
    every later gap has all 4 axis neighbours). `pass_pixels(pass)` is the
    single source of truth (both backends use it), `pass_of(r, c)` its inverse.
  - **Tiles store escape iterations, not colours** (0 = in set, n = escaped
    at n; bit 31 = not computed). Changing iterations calls `Tile::retarget`:
    ↓ is exact with no recompute (n > max → in set); ↑ clears only in-set
    pixels, lowers `passes_done` to the still-complete prefix and keeps
    displaying at the old level (`display_floor`, cleared pixels drawn black).
    `Tile::version` changes whenever displayable contents change.
  - **Quadtree seeding** (`seed_from_relatives`): pixel (i,j) at depth d is
    pixel (2i,2j) at depth d+1, so a finished parent gives a child passes
    0–3 for free, and four children give their parent. Same iterations only.
- `src/tiles/render.rs` — `run_generation`: lists visible tiles, retargets and
  seeds them, sorts them by distance from the cursor, and for each pass calls
  `backend.render_pass_batch(ctx, &batch, pass, &int)` once with every tile that
  still needs that pass. Owns `GroupCache` (the CPU's per-group reference orbits).
- `src/tiles/perturb/mod.rs` — the `Perturbator` trait (`render_pass_batch`),
  `PassBatchCtx` (incl. the store's `progress` counter), `TileItem`, `Toggle`.
- `src/tiles/perturb/cpu.rs` — CPU backend. Runs tiles through `rayon` `par_iter`
  (`DETERMINISTIC` const = sequential). A glitched pixel computes its own exact
  orbit and pushes it onto the group `RefList`. Also does black-fill: if pass 0
  plus the tile perimeter are all black, the whole tile is filled black.
- `src/tiles/perturb/gpu.rs` — GPU backend (details below). Its test module
  holds the GPU harnesses and regression tests (see Testing).
- `src/tiles/perturb/nucleus.rs` — reference search: ball-period detection +
  Newton to a nucleus, z in FBig, dz/dc as f64 mantissa + exponent (`Deriv`).
- `src/tiles/perturb/bla.rs` — BLA table builder (CPU floatexp `Fx`).
- `src/tiles/perturb/shaders/` — `floatexp.wgsl` (helpers), `perturb_common.wgsl`
  (BLA lookup/apply, jump limits, interior check), `perturbation.wgsl` (f32
  body), `perturbation_floatexp.wgsl` (deep body). Assembled by `shader_source`.
- `src/drawing/renderer/multithreaded.rs` — the compute thread: one async
  loop (`compute_loop`) that makes the backends (`Toggle`: CPU/GPU) and
  awaits `run_generation` per requested view.
- `src/gpu_compositor.rs` — draws the tiles to screen (its own wgpu device,
  separate from the compute backend) on the render thread.
  - **Colouring on the GPU**: a tile's iteration counts are uploaded
    (`iteration_texels`: the stride grid or the whole tile; 0 = black, incl.
    pixels being recomputed after ↑; `NONE_RAW` = sub-pass gap) when its
    `version` changes, and a compute pass (`gpu_compositor_recolour.wgsl`)
    makes the colour texture's drawn mip levels: palette, gap fill (mean of
    the computed axis neighbours) and mip averaging, all in linear light. A
    palette change (`palette_gen`) re-runs only that pass. The palette is
    **computed in the shader** (`colour`, a 16-byte uniform of the phases);
    `rendering::val_to_color` is the test-only reference, kept in sync by
    `recolour_matches_cpu` (change both). A table lookup was the pass's
    main cost on noisy tiles (a gather by iteration).
  - **Chunks**: tiles of one `Shape` (data size, drawn levels) share 2D
    array textures (`Chunk`, 16…256 layers, a slot per tile, dropped once
    empty): colouring is one dispatch per chunk and level (layer lists via
    a dynamic offset), drawing one draw call per chunk. Metal serialises
    dispatches (~10–15 µs each), so per-tile dispatches took 365 ms for a
    recolour at s = 4. `diag_recolour_speed` (M2 Pro, full 3000×2000 screen):
    s = 4 (24000 tiles) 25–30 ms, s = 1 ~10 ms. RAM: the iterations stay
    on the GPU (64 KiB per full tile; ~1.5 GiB at s = 4), accepted by the
    user for fast recolouring.
  - **Sampling and antialiasing**: tiles are rendered at the depth where a
    screen pixel spans r ∈ [s, 2s) tile pixels per axis (`TileStore::
    min_ratio` / `depth_for_view`; s shared with the compute thread through
    the store). Tile colour textures are sampled as sRGB, each level the
    area average in linear light of the data texels it covers (colouring
    pass), holding only the one or
    two levels drawn at the current s (`drawn_levels`: level 0 at s ≤ 1,
    level 2 at s = 4, i.e. 1/16 of the texels; finer per use for parent
    previews); a change of s changes the tiles' shapes, so re-uploads and
    recolours. Tiles are drawn 1:1 into an offscreen
    sRGB image (≤ 2× the surface per axis) from the level k that leaves
    q = r/2^k ∈ [1, 2) texels per screen pixel, grid-aligned; the downsample
    pass (`gpu_compositor_downsample.wgsl`) takes each screen pixel's exact
    q×q area average (bilinear below q = 1) and encodes to sRGB (the surface
    is not sRGB). Cost vs plain bilinear (startup view, 3000×2000): ~+2 ms
    compositor GPU per frame (median 2.9 → 4.8 ms), compute ~3–5% slower
    from sharing the GPU.
  - The progress bar: a white fill proportional to
  the pixel work done (`BarAnim`): rising, it moves at the measured progress
  speed (so updates seconds apart don't show as jumps), catching up over
  about one update interval when far behind; falling (restart), a critically
  damped spring; never ahead of the real progress. Rising has momentum and
  trails the progress by ~2 update intervals, so irregular updates do not
  make it jerk. `render()` returns
  whether the bar is still moving, and the render thread keeps drawing at
  vsync until it settles.

## Web build (`web/`, `platform.rs`)

The same code compiled to wasm on WebGPU (`web/build.sh` → `web/dist`,
`web/serve.py` serves it with the headers; Cloudflare Pages via `_headers`).
Checked in headless Chrome driven over CDP (render, zoom, palette, s, the
ceiling); deep zoom (floatexp pipeline, BLA, nucleus search at depth) is
not yet checked in a browser.

- **Threads are Web Workers sharing one wasm memory** (`wasm_thread`, with
  our own worker script: theirs closes the worker when the thread function
  returns). Needs cross-origin isolation (COOP/COEP headers), nightly with
  `-Z build-std` and the flags in `.cargo/config.toml` (atomics, shared
  imported memory up to 4 GiB, TLS exports; current nightlies no longer add
  the shared-memory link args for `+atomics`). parking_lot needs its
  `nightly` feature there, or parking panics.
- **The compute path is async** (`BatchFuture`, `run_generation`, the GPU
  dispatch chain): a worker can't block on a GPU readback (the mapping
  resolves from its event loop). Natively the compute thread drives it with
  `pollster` and `wait` still blocks on `device.poll`, so nothing changed
  (checked: same timing). GPU objects never leave their thread (on the web
  they are `!Send`), so the compute loop makes its own `GpuState`.
- **The browser's main thread must never block** (a contended lock or a
  wait throws): the compositor draws there (`WebRenderer`, on each
  animation frame, instead of the render thread), with no rayon
  (`platform::ui_map`); the compute worker, the search and rayon's pool run
  in workers. The main thread still takes the view channel's lock briefly
  (`waker_interrupter::Sender::send`); contention is rare, but a true fix
  would be a lock-free send.
- The window is the page's `<canvas id="canvas">`; its size comes from the
  layout until winit's resize observer fires (`initial_size`).
- Memory: 3 GiB ceiling for the tiles (see the tile budget); native has none.

## Perturbation math

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

**Rebasing** (Zhuoran 2021, GPU): when |z| < |δ|, set δ ← z² + δ₀ and restart
the reference index at 0 (the X₀ = C form of "δ ← z, back to the start"). It
is exact, not a heuristic, and removes the blocky precision-loss artifacts
that appear once the reference is a nucleus the pixel's orbit does not follow.
The shaders track `n` (iterations, reported) and `m` (reference index)
separately.

**BLA** (bilinear approximation, GPU; `bla.rs` + `perturb_common.wgsl`): while
|δ| is tiny next to |2X| a step is linear, δ ← A·δ + B·δ₀, and blocks of 2^k
steps compose into one (A, B) with a validity radius R (relative error ≤ 2⁻²⁴):
`A = A_y·A_x, B = A_y·B_x + B_y, R = min(R_x, (R_y − |B_x|·max|δ₀|)/|A_x|)`,
single step `R = ε·|2X|`. The table is a binary tree over the reference orbit
(level k: blocks at multiples of 2^k, level k at index `2^(L+1) − 2^(L+1−k)`),
built on the CPU in a small floatexp (`Fx`; A overflows and R underflows f64 at
depth) per pass and reference, with R bounded by the pass's max |δ₀|. The
shaders search upwards from level 1 (a block's R ≤ its first half's, so the
first invalid level ends the search; level 0 is never used) with exponential
back-off after failed searches (reset on a jump or rebase), apply the block in
floatexp, add log|A|² to the interior-detection derivative, and never jump
across an interior window boundary or the iteration limit. No escape can be
skipped: within R, |z| ≤ |X|·(1 + 2⁻²³), and single steps with |X| near or
above 2 (an escaping reference's end) are invalid, so no block spans them.
log2|δ| is clamped above the invalid radius (at δ = 0 it is −∞, which passed
invalid blocks: a pixel at its own reference jumped over its escape).
`USE_BLA` switches it off for A/B. Measured: the 2⁻³⁰⁵ reported view went from
~2 min to ~1.5 s per generation and became more accurate (fewer f32 rounding
steps); shallow views ~5–10% slower.

**Interior detection** (GPU, nucleus references only): the shaders track
log2|dz/dz₀|² and, every window of ≥128 iterations (a whole number of the
nucleus period p), compare it with the previous window; 2 consecutive windows
contracting by ≥0.9× per period → in the set. Inside a component the cycle
multiplier |λ| < 1; just outside |λ| ≥ 1. Short windows (< ~64) produce false
positives (escaping pixels contracting briefly near 0), so don't shorten them.
Parameters are the `INTERIOR_*` consts in gpu.rs, passed via uniforms.

## GPU backend (`gpu.rs`)

- **Reference = a nearby nucleus** (`nucleus.rs`). Its orbit never escapes,
  so normally nothing glitches. Seeded from the view centre; if the centre
  escapes too early to reveal the period, glitch rounds evaluate 32 sampled
  glitched pixels exactly (in parallel), take the longest-lived, and seed the
  search from it. The best reference is cached in `GpuState` across passes and
  views: a full one (nucleus) is reused up to `MAX_REF_REUSE_DIST` (1024)
  view radii away, so zooming rarely searches (measured, `diag_far_reference`:
  3500 radii as accurate as 13; a view's own deeper nucleus is more accurate
  still). Not much further: pixel offsets have a 24-bit mantissa, so pixel
  positions are quantized to ~distance·2⁻²⁴ (4 px at 2²⁶ px: 26 of 600 wrong
  vs 0; views further out only looked fine because they were nearly
  uniform). A searched-for nucleus must lie within
  `MAX_REF_DIST` (16) radii. Failed centre and glitch-seeded
  searches (incl. a nucleus whose orbit escapes) are remembered for views
  whose centre lies in the failed view with a radius within 2×
  (`ViewGeom::covered_by`).
- **Nucleus precision**: a component is only ~1/|dz_p/dc|² wide (2⁻⁶³⁰ at a
  2⁻³²⁵ view, p≈49000), far below the view's precision; Newton converges at
  the view's precision, then raises it to 2·log2|dz_p/dc| + 64 bits and
  converges into the component. Without that the "nucleus" orbit escaped
  after ~1.1 periods. Far from the root Newton walks with a constant step;
  same-direction steps are multiplied (doubling) to get there sooner. At
  2⁻³¹⁶…2⁻³³⁰ a search takes 0.7–3.6 s.
- **Background search** (`SearchJob`, `start_search`, `search_result`): when
  no cached nucleus is within reach, the centre search runs in a background
  thread (keyed by view like the failed-search memory; a search for another
  view cancels it) and the pass starts at once on a provisional reference,
  the view centre's orbit. Pixels that escape before it are exact and
  stored immediately; glitched ones are **deferred** (not stored, no glitch
  rounds). Between chunks the pass polls the search and switches to the
  nucleus for later chunks; at the end of the pass it waits for the search
  (interruptibly) and redoes the deferred pixels against the nucleus, or,
  if none was found, sends them through the usual glitch rounds. After a
  paste at 2⁻³¹⁴…2⁻³²⁶: first pixels after ~110 ms instead of 1–2 s,
  accuracy as with the view's own nucleus. Total time per generation
  (4 interleaved runs vs master): unchanged where a nucleus is found (pass 0
  still ends waiting for it: 3.4–3.6 s), −21% where none is (2⁻³¹⁸:
  6.9 → 5.4 s; the deferred pixels go through glitch rounds once instead of
  per chunk). A generation that started on the
  provisional reference can differ from a later one by f32 rounding in a
  few pixels (1 of 15000 at 2⁻³⁰⁵).
- References need not lie on the pixel grid: `ref_px` gives the reference's
  position in pixel units, computed exactly in FBig; pixel offsets are
  `col - ref_px` for both pipelines.
- **`GpuRef`**: a reference prepared for dispatching — orbit and BLA table
  uploaded once per pass (and per glitch-round reference), plus uniform values.
  The dispatch chain (`dispatch_offsets` → `dispatch_f32/fe` → `dispatch` →
  `dispatch_chunk` → `dispatch_once`) takes it.
- **Chunks**: each pass's pixels are collected in cursor order and dispatched
  in chunks of ~`TARGET_CHUNK_MS` (30 ms): a `PROBE_CHUNK_PX` (2048) probe
  first, then sized from the last chunk's measured ms/pixel, floor
  `MIN_CHUNK_PX` (1024; each dispatch has ~0.8 ms fixed overhead). The
  interrupt is checked between chunks; in-flight GPU work cannot be cancelled.
  Per chunk: dispatch → glitch rounds (up to `MAX_GLITCH_PASSES=8`) → residual
  CPU resolve → store → finish every tile whose pixels are all stored → bump
  `ctx.progress` so the compositor redraws.
- **CPU/GPU overlap**: one chunk on the GPU at a time, but the next chunk's
  seeds are packed (`Deltas::pack`) while waiting, and it is submitted
  (`submit_deltas`) as soon as the current one is read back (`finish`),
  before the current one's glitch rounds, resolve and storing. ~17% faster
  per generation at the 2⁻³⁰⁵ view (the GPU was idle ~30% of a pass). Don't
  queue two chunks at once: Metal runs them concurrently, each ~2× slower per
  pixel, and their times can't be measured, so chunk sizing broke (a 1.5M px,
  148 ms dispatch). The next chunk is sized from the chunk before the current
  one (except right after the probe).
- **Keep dispatches short: macOS kills long GPU command buffers** (seen at a
  few hundred ms) and then drops some following ones, and wgpu reports
  neither; the output would silently keep its initial value. So besides the
  time-based sizing, the output buffer is pre-filled with `NOT_RUN`
  (u32::MAX, never written by the shaders); any `NOT_RUN` left is retried in
  halves (down to `MIN_RETRY_PX`), and pixels still not run are not stored.
  Symptom when this breaks: black or partly black tiles that change on every
  reset.
- Output encoding: `0` = in set, `n` = escaped at iteration n,
  `GLITCH_BIT (0x8000_0000) | n` = glitched, `NOT_RUN` = not computed.
- **Never cache a guess**: a pixel that still carries `GLITCH_BIT` (or
  `NOT_RUN`) is *not stored*, so the next generation retries it. Leftover
  glitches are resolved exactly on the CPU (`calculate_orbit` + `check_orbit`),
  in parallel and interruptibly. Storing glitches as black and then skipping
  them (`is_some()`) across interrupted generations caused wrong-iteration
  pixels mixed into correct ones while zooming.
- A tile's `finish_pass` runs as soon as every pixel it needed this pass is
  stored (even mid-pass or when interrupted), and only if it has finished the
  previous pass. A tile with an unstored pixel never finishes, so it is retried.
- **2D dispatch**: `gx=min(groups,65535)`, `gy=ceil(groups/65535)`. The shader
  rebuilds `idx = gid.y*dispatch_w + gid.x`, because a dimension may not exceed 65535.
- **Binding-size chunking**: `dispatch()` also splits batches so the delta
  buffer stays under `max_storage_buffer_binding_size` (128 MiB here). It takes
  an `elem` size: 8 B for f32 deltas, 16 B for floatexp seeds.
- **Host/WGSL layouts must match**: `Uniforms` (48 bytes) and `BlaEntry` /
  `struct Bla` (32 bytes, vec2 fields first since WGSL aligns vec2<f32> to 8).
  wgpu only rejects a mismatch at dispatch time, on a GPU;
  `entry_layout_matches_wgsl` pins the BLA one.
- The black-fill optimisation is intentionally absent on the GPU.

### Precision / floatexp

- f32 hits a hard wall at the **exponent**, not the mantissa. Once
  units-per-pixel falls below ~2⁻¹²⁶ (≈1e-38) the seed δ₀ is subnormal, and
  neighbouring pixels collapse to one value. Double-single (hi+lo f32) would
  **not** help with that, because it keeps the f32 exponent range.
- Fix: shared-exponent complex **floatexp** (`Fe { m: vec2<f32>, e: i32 }`, in
  `shaders/floatexp.wgsl`). The deep shader works in two phases: δ in floatexp
  until `e > -60`, then plain f32 (δ₀ is negligible there). **δ can become
  tiny again**: at a zero of the reference orbit (a nucleus orbit hits 0 once
  per period, where 2·X·δ vanishes and δ ← δ² + δ₀), after a rebase, or after
  a BLA jump. Such a phase-2 step (result below 2⁻⁶²) is redone in floatexp
  and the pixel returns to phase 1. Switching at 2⁻¹⁰⁰ without that fallback
  let δ underflow to exactly 0, and pixels followed the reference forever
  (black octagons and smeared streaks at 2⁻²²⁰). With the fallback, switch
  thresholds 2⁻⁶⁰…2⁻¹¹⁰ are equally accurate and 2⁻⁶⁰ is fastest.
- The pipeline is chosen per pass: `use_fe = upp_log2(depth) < FE_THRESHOLD (-100)`.
- `pack_fe(dx, dy, base_exp)`: seeds are built in **pixel units** (O(1e4), so the
  f64 math is always safe), and the depth scale `2^upp` goes into the exponent.
  Never form `2^upp` as an f64 on the deep path, because it underflows.
- The shaders are compiled by Metal with **fast math** (wgpu does not turn it
  off): don't rely on infinities or NaN semantics (e.g. BLA's invalid radius is
  a finite `-1e30`, not −∞).
- The reference orbit is uploaded as plain f32 for the f32 pipeline and as
  **floatexp** (`Reference::orbit_fe`, `pack_fe` layout) for the deep one;
  `prepare(r, log2_dc, upp)` picks the format. A deep orbit passes far
  closer to 0 than f32 reaches (a nucleus comes within ~2⁻¹⁴⁹ / 2⁻²⁷¹ of 0 at
  its parents' periods, at 2⁻³¹⁴). Flushed to 0 there, the 2·X·δ term
  vanished while δ was smaller still, so every pixel shadowed the reference:
  all black with a nucleus, or all "escaping" with a point reference
  (`view_2026_09_23c_matches_exact`). Both `orbit_fe` and the BLA table come
  from an f64 copy (`Reference::orbit64`), which limits them to ~2⁻¹⁰²².
- **Mantissa precision is the remaining accuracy limit** (`diag_reference_precision`,
  ignored): f32 rounding makes a fraction of long-lived near-boundary pixels
  escape a few iterations off (at the 2⁻³⁰⁵ view: 611 of 15000 samples off,
  10 by >50 iterations, with BLA; 5946/112 without). Mostly from rounding the
  reference orbit to f32, but a hi+lo reference alone does not fix it.

## Testing

- `cargo test` runs the regular tests (no GPU needed): pass layout, seeding
  geometry, retarget, nucleus search, BLA table (incl. jump = stepping), the
  CPU mirror of the reference path, interior detection safety
  (`interior_detection_is_safe_and_useful`), and `shaders_validate` (parses and
  validates the WGSL with naga, since it is otherwise only compiled when the
  GPU backend starts).
- `cargo test --release -- --ignored` runs GPU tests and measurements on this
  machine's GPU against exact FBig orbits, at the views the user reported
  (2026-09-23): `artifact_view_2026_09_23_matches_exact` (2⁻²²⁰),
  `view_2026_09_23b_pipeline_matches_exact` (2⁻³⁰⁵, whole pipeline incl.
  resets), `view_2026_09_23c_matches_exact` (2⁻³¹⁴, 262144 iterations),
  `dispatch_is_deterministic_even_when_killed`,
  `bla_with_escaping_reference_matches_plain`,
  `zero_delta_escapes_with_reference`, `recolour_matches_cpu` (compositor
  colouring vs `val_to_color`), plus `diag_*` measurements
  (`diag_recolour_speed`: a full-screen recolour, `DIAG_TILES`,
  `DIAG_BASE`, `DIAG_SMOOTH`)
  (`diag_bla_ab`, `diag_switch_threshold`, `diag_interior_detection`,
  `diag_reference_precision`, `diag_far_reference`: accuracy with a reused
  far-away nucleus, `diag_deep_nucleus_search`: search timing
  zooming below the 2⁻³¹⁴ view, `DIAG_LEVELS=-316,-320`, `DIAG_PIPE=1` for
  whole generations). Set `DIAG_OUT=dir`
  to cache exact values (the 2⁻³⁰⁵ ones take ~1 min) and dump renders as raw
  RGB.
- **Visual bug with coordinates → reproduce first**: parse the Cmd-C string
  with `sample_view`, run the real shader (`gpu_on_sample`) or the whole
  pipeline (`run_pipeline`), score against exact orbits (`score`), and A/B
  fixes there, rather than theorising or trusting a CPU mirror of a shader.

### Conventions

- WGSL lives in `.wgsl` files and is loaded with `include_str!`; a shader is
  `shader_source(body)` = floatexp helpers + `perturb_common.wgsl` + body.
  Keep shader code out of Rust string literals.
- `Pf` in `rendering.rs` is currently **`f32`**. That was set on purpose so the
  CPU matches GPU precision for comparison testing. The CPU's normal setting is
  `f64`, which reaches far deeper. Ask before changing it.
- Commit each logical step separately.

## Open threads (as of 2026-09-24)

Done: bounded chunked dispatch, nucleus references, rebasing, sub-passes,
quadtree seeding, iteration retargeting, interior detection, BLA, the floatexp
underflow, dropped-dispatch and floatexp-orbit fixes, deep nucleus search
(precision, speed, caching, far reuse), BLA near an escaping reference, the
momentum progress bar, CPU/GPU overlap, background nucleus search, GPU as
the default backend, the sampling ratio s with area-averaging
antialiasing (incl. a tile budget that follows the view), palette
scrolling, GPU recolouring, and the web build (see Web build). Decided: render order stays pass-major (each
pass completes, rippling out from the cursor, before the next starts); the
user rejected ring-by-ring refinement. Still open (talk through before
coding):

1. **Share one wgpu device** between the compositor and compute (the CPU/GPU
   overlap half of this item is done).
2. **Compute fewer pixels**: done as the sampling ratio s (0.5 = at most
   1:1, ~¼–1× the screen's pixels, vs 1–4× at s = 1). Note only (the user
   doesn't care for now): the offscreen image is always 2× the surface per
   axis, but is fully used only just before a depth switch; sizing it to
   the ratio in use would save ~1 ms of GPU bandwidth per frame. It is
   scratch space, so this is no extra computation.
3. **Mantissa precision**: double-single delta arithmetic, or accept f32
   (much better with BLA). Undecided.
4. **CPU backend** lacks the GPU's nucleus references, rebasing, BLA and
   interior detection; the GPU lacks the CPU's black-fill. Not solid
   guessing (fill a cell whose corners match): the user said not yet.
5. **Compositor draw calls at high s**: done with the chunks (one draw call
   per chunk, see the compositor). Still to measure: whether compute
   chunks shrink to the minimum at s = 4 ("slow af" is mostly the 16–64×
   work). Not done: freeing mostly-empty chunks (a chunk is dropped only
   when entirely empty, so eviction can leave sparse ones).
6. **Don't wait for the search at the end of pass 0** (optional, only if the
   ~1–2 s wait still bothers the user): move on to later passes while
   deferred pixels are pending. Tricky: a tile finishes a pass only once all
   its pixels are stored, and later passes would defer more pixels too.
7. **TODO (user): main-thread lock in the view channel (web).**
   `waker_interrupter::Sender::send` (called on the browser's main thread on
   every view change) takes the channel's parking_lot mutex, which the
   compute worker also holds briefly (every `interrupted()` check between
   chunks, and while waiting in `recv_multithreaded`). If contended, parking
   calls `Atomics.wait`, which throws on the main thread; `notify_one` can
   too (parking_lot's internal bucket lock). Rare (ns-long holds) but
   possible, e.g. zooming during a render. Fix in waker_interrupter: a
   lock-free send (store the message in an atomic slot, wake the receiver
   without blocking) and an atomic-flag interrupter check.
