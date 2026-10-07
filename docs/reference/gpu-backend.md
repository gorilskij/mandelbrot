# The GPU backend

`src/tiles/perturb/gpu.rs` (with `nucleus.rs`, `bla.rs` and `shaders/`): the main backend,
and the default at start-up. The math it implements: [perturbation](../explanation/perturbation.md).
Its tests: [testing](testing.md).

## References

- **The reference is a nearby nucleus** (`nucleus.rs`): ball-period detection, then Newton to
  the nucleus (z in `FBig`, dz/dc as an f64 mantissa + exponent, `Deriv`). A nucleus orbit
  never escapes, so normally nothing glitches.
- **Seeding the search:** from the view centre; if the centre escapes too early to reveal the
  period, glitch rounds evaluate `GLITCH_CANDIDATES` = 32 sampled glitched pixels exactly (in parallel), take the
  longest-lived, and seed the search from it.
- **Caching:** the best reference is kept in `GpuState` across passes and views. A full one
  (a nucleus) is reused up to `MAX_REF_REUSE_DIST` = 1024 view radii away, so zooming rarely
  searches (measured with `diag_far_reference`: 3500 radii as accurate as 13; a view's own
  deeper nucleus is more accurate still). Not much further: pixel offsets have a 24-bit
  mantissa, so pixel positions are quantized to ~distance·2⁻²⁴ (4 px at 2²⁶ px: 26 of 600
  wrong vs 0).
- A searched-for nucleus must lie within `MAX_REF_DIST` = 16 view radii (`nucleus.rs`).
- **Failed searches are remembered** (centre and glitch-seeded, including a "nucleus" whose
  orbit escapes) for views whose centre lies in the failed view with a radius within 2×
  (`ViewGeom::covered_by`). Space forgets them.
- **Nucleus precision:** a component is only ~1/|dz_p/dc|² wide (2⁻⁶³⁰ at a 2⁻³²⁵ view,
  p ≈ 49000), far below the view's precision. Newton converges at the view's precision, then
  raises it to 2·log2|dz_p/dc| + 64 bits and converges into the component (without that the
  "nucleus" orbit escaped after ~1.1 periods). Far from the root Newton walks with a constant
  step; same-direction steps are doubled to get there sooner. At 2⁻³¹⁶…2⁻³³⁰ a search takes
  0.7–3.6 s.
- References need not lie on the pixel grid: `ref_px` is the reference's position in pixel
  units, computed exactly in `FBig`; pixel offsets are `col − ref_px` in both pipelines.

### Background search

When no cached nucleus is within reach, the centre search runs on a background thread
(`SearchJob`, `start_search`, `search_result`; keyed by view like the failed-search memory;
a search for another view cancels it), and the pass starts at once on a **provisional
reference**, the view centre's orbit:

- pixels that escape before it are exact and stored immediately;
- glitched ones are **deferred** (not stored, no glitch rounds);
- between chunks the pass polls the search and switches to the nucleus for later chunks;
- at the end of the pass it waits (interruptibly) and redoes the deferred pixels against the
  nucleus — or, if none was found, sends them through the usual glitch rounds.

After a paste at 2⁻³¹⁴…2⁻³²⁶: first pixels after ~110 ms instead of 1–2 s, accuracy as with
the view's own nucleus. A generation begun on the provisional reference can differ from a
later one by f32 rounding in a few pixels (1 of 15000 at 2⁻³⁰⁵).

## Dispatching

- **`GpuRef`**: a reference prepared for dispatching — its orbit and BLA table uploaded once
  per pass (and per glitch-round reference), plus the uniform values.
- **Two paths take it.** The pass's main chunks: `Deltas::pack` (seeds as f32 or floatexp)
  → `submit_deltas` (split under the binding size, submitted without waiting) → `finish`
  (read back, `retry_not_run`). Everything else (glitch rounds, tests): `dispatch_offsets`
  → `dispatch` → `dispatch_chunk` → `dispatch_once`. `pipeline_for` picks the f32 or the
  floatexp pipeline from the packed seeds.
- **Pipeline per pass:** floatexp when `upp_log2(depth) < FE_THRESHOLD` (−100), else f32
  ([precision](precision.md)).
- **Chunks:** a pass's pixels are collected in cursor order and dispatched in chunks of about
  `TARGET_CHUNK_MS` = 30 ms: a `PROBE_CHUNK_PX` = 2048 probe first, then sized from the last
  measured ms per pixel, floor `MIN_CHUNK_PX` = 1024. The interrupt is checked between chunks;
  in-flight GPU work cannot be cancelled.
- **The cost per pixel** (`ms_per_px`): where the device has `TIMESTAMP_QUERY` (requested when
  available), each dispatch's compute-pass time from a begin/end timestamp pair plus
  `DISPATCH_OVERHEAD_MS` = 0.8 (without it a chunk of cheap pixels would make the next one
  huge), set in `wait`. The pairs rotate over `TIMESTAMP_SLOTS` = 16: reusing one made Firefox
  report the previous dispatch's. Otherwise a chunk's submit → readback time (`record_chunk`),
  where any readback delay counts as cost: a fixed delay a settles chunks at b·n = 30 ms − a,
  and at a ≥ 30 ms they collapse to the floor ([web build](web-build.md#firefox-reads-back-late)).
- **Waiting for a readback on the web** (`wait`): `firefox_nudge` submits an empty command
  buffer every 4 ms from shortly before the estimated completion (`ms_per_px` × pixels, kept in
  `InFlight::expected`) until the result arrives. Natively `wait` blocks in `device.poll`.
- **Diagnostics per dispatch** (the `[diag gpu] dispatch` line): submit → readback time, the
  estimate, nudges, `prep` (buffer creation and encoding before the submit) and `gpu` (the
  timestamps, where available).
- **Per chunk:** dispatch → glitch rounds (up to `MAX_GLITCH_PASSES` = 8, each against a
  better reference) → residual exact resolve on the CPU (`calculate_orbit` + `check_orbit`,
  in parallel, interruptible) → store → finish every tile whose pixels are all stored → bump
  `ctx.progress` so the compositor redraws.
- **CPU/GPU overlap:** one chunk on the GPU at a time, but the next chunk's seeds are packed
  (`Deltas::pack`) while waiting, and submitted (`submit_deltas`) as soon as the current one
  is read back, before its glitch rounds, resolve and storing. ~17 % faster per generation at
  the 2⁻³⁰⁵ view (the GPU had been idle ~30 % of a pass). **Never queue two chunks at once**:
  Metal runs them concurrently, each ~2× slower per pixel, and their times cannot be measured,
  so sizing broke (a 1.5 M px, 148 ms dispatch). The next chunk is sized from the chunk before
  the current one (except right after the probe).
- **Keep dispatches short — macOS kills long command buffers** (seen at a few hundred ms) and
  then drops some following ones, and wgpu reports neither: the output would silently keep
  its initial value. So the output buffer is pre-filled with `NOT_RUN` (`u32::MAX`, never
  written by the shaders); any `NOT_RUN` left is retried in halves down to `MIN_RETRY_PX` =
  256, and pixels still not run are not stored. Symptom when this breaks: black or partly
  black tiles that change on every reset.
- **2D dispatch:** `gx = min(groups, 65535)`, `gy = ceil(groups / 65535)`; the shader rebuilds
  `idx = gid.y · dispatch_w + gid.x` (a dimension may not exceed 65535).
- **Binding-size chunking:** `dispatch()` also splits batches so the delta buffer stays under
  `max_storage_buffer_binding_size` (128 MiB here); element size 8 B (f32 deltas) or 16 B
  (floatexp seeds).

## Results

- **Output encoding:** `0` = in the set, `n` = escaped at iteration n,
  `GLITCH_BIT (0x8000_0000) | n` = glitched, `NOT_RUN` = not computed.
- **Never cache a guess:** a pixel still carrying `GLITCH_BIT` or `NOT_RUN` is **not stored**,
  so the next generation retries it. (Storing glitches as black and skipping them across
  interrupted generations once mixed wrong-iteration pixels into correct ones while zooming.)
- A tile's `finish_pass` runs as soon as every pixel it needed this pass is stored (even
  mid-pass or when interrupted), and only if it finished the previous pass. A tile with an
  unstored pixel never finishes, so it is retried.
- **No black-fill** on the GPU (the CPU backend's shortcut), on purpose.

## Shaders

`src/tiles/perturb/shaders/`: `floatexp.wgsl` (floatexp helpers), `perturb_common.wgsl`
(BLA lookup and apply, jump limits, interior check), `perturbation.wgsl` (the f32 body),
`perturbation_floatexp.wgsl` (the deep body). A shader is
`shader_source(body)` = floatexp helpers + `perturb_common.wgsl` + body, loaded with
`include_str!` — **keep WGSL out of Rust string literals**.

- **Host and WGSL layouts must match:** `Uniforms` (48 bytes) and `BlaEntry` / `struct Bla`
  (32 bytes, vec2 fields first since WGSL aligns `vec2<f32>` to 8). wgpu rejects a mismatch
  only at dispatch time, on a GPU; `entry_layout_matches_wgsl` pins the BLA one.
- `shaders_validate` parses and validates all WGSL with naga in `cargo test` (otherwise it is
  compiled only when the GPU backend starts).
- Metal compiles with **fast math** (wgpu does not turn it off): no reliance on infinities or
  NaN (BLA's invalid radius is a finite `−1e30`, not −∞).

## Tuning constants

In `gpu.rs` unless noted: `USE_BLA` (true; switch off for A/B), `MAX_GLITCH_PASSES` = 8,
`TARGET_CHUNK_MS` = 30, `PROBE_CHUNK_PX` = 2048, `MIN_CHUNK_PX` = 1024, `MIN_RETRY_PX` = 256,
`MAX_REF_REUSE_DIST` = 1024, `FE_THRESHOLD` = −100, `INTERIOR_MIN_WINDOW` = 128,
`INTERIOR_WINDOWS` = 2, `INTERIOR_Q` = 0.9, `INTERIOR_RETURN` = 2⁻²⁰ (on squared magnitudes:
2⁻¹⁰ relative), and `MAX_REF_DIST` = 16 (`nucleus.rs`).
