# Pitfalls

Traps this project has fallen into, or nearly. Each: the symptom, the cause, what to do.

## The math

- **Do not add a Pauldelbrot-style |δ|/|X| glitch test.** Perturbation is exact for any size
  of δ; a large δ only risks float precision. An `|δ|² > 1e−6·|x|²` check in the shader flagged
  nearly every pixel at low zoom and drew an "iteration-1 egg". A pixel is glitched only when
  the reference orbit ended (`!is_full`) before it escaped.
- **Do not shorten interior detection's windows** (≥ 128 iterations): short windows (< ~64)
  make escaping pixels that contract briefly near 0 look interior.
- **Interior detection with a foreign nucleus** (a cached one of another period) put windows
  out of phase with the pixel's cycle: black discs and misshapen minibrots. The return test
  (`INTERIOR_RETURN`) fixed it; do not remove it.

## The GPU

- **macOS kills long GPU command buffers** (a few hundred ms) and drops some following ones;
  wgpu reports neither. Keep dispatches short (~30 ms chunks), keep the `NOT_RUN` sentinel and
  its retry. Symptom: black or partly black tiles that change on every reset.
- **Never queue two chunks at once**: Metal runs them concurrently, both slow down and chunk
  sizing breaks.
- **Never store a guess**: a pixel with `GLITCH_BIT` or `NOT_RUN` must not be stored, or
  interrupted generations mix wrong pixels into right ones.
- **Host and WGSL struct layouts must match** (`Uniforms` 48 B, `BlaEntry` 32 B, vec2 fields
  first). wgpu only notices at dispatch, on a GPU. Run `cargo test` (`shaders_validate`,
  `entry_layout_matches_wgsl`).
- **Metal compiles shaders with fast math**: no infinities, no NaN tricks.
- **Never form 2^upp as an f64 on the deep path**: it underflows. Build seeds in pixel units.
- **A tiny δ in the f32 phase underflows to 0** and the pixel follows the reference forever
  (black octagons, streaks): keep the fallback to floatexp below 2⁻⁶².
- **Flushing a deep orbit to f32** zeroes 2·X·δ: upload the floatexp orbit for the deep
  pipeline.
- **The palette exists twice** (the WGSL `colour` and `rendering::val_to_color`): change both;
  `recolour_matches_cpu` checks it.

## The web

- **The browser's main thread must never block**: no contended lock, no wait, no rayon there.
- **GPU objects are `!Send` on the web**: each thread makes its own.
- **The wasm-bindgen CLI must match the library exactly**: keep `Cargo.lock` committed;
  `build.sh` installs the matching CLI.
- **The release profile keeps debug info**: a deployed build must turn it off (static assets
  reject files over 25 MiB). `cf-build.sh` does.

## Process

- **Visual bug with coordinates → reproduce it first** in the test harness with the exact
  view, then A/B fixes there, rather than theorising or trusting a CPU mirror of a shader
  ([how](../how-to/reproduce-a-visual-bug.md)).
- **`Pf` is f32 on purpose**: ask before changing it.
- **Deploy from `test-website` / `pub-website`, never from `master`.** On 2026-10-02 both
  games' test Workers were first deployed from `master`-based branches; here `master` differed
  only in docs, but for hex snake it shipped unreleased work. `master` can carry ongoing work
  ([deploy](../how-to/deploy.md)).
- Commit each logical step separately.
