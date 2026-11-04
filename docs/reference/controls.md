# Controls

Input handling is in `src/main.rs` (`App::window_event`). The same controls work natively
and on the web.

| input | does |
|---|---|
| scroll | zoom around the cursor: each wheel notch (a trackpad's ~10 px) scales the view by 2.5 %, at most 20 % per event (`apply_scroll_zoom`). The zoomed view shows at once, interpolated from existing tiles, and is recomputed around the cursor. Zooming out and panning stop at the start-up view's rectangle (`bounds`, `clamp_to_bounds`) |
| drag (left button) | pan |
| **Cmd**-scroll | shift the palette phases that are toggled on (below): hue 1/128 turn per notch, lightness 1/32 (recolours only, nothing recomputed) |
| tap **Alt** / **Ctrl** / **Shift** | toggle which phases the colour controls move: Alt the **hue**, Ctrl the **lightness**, Shift **inverts** the lightness direction (so hue and lightness can go opposite ways). A tap is the modifier pressed and released with no other key, click or scroll in between; in a chord (Shift+`]`, Ctrl-C) it toggles nothing. At start: hue on, lightness off, not inverted |
| **`]` / `[`** | set the palette **drift** speed one step up / down the ladder …, −1.5×, −1×, 0, +1×, +1.5×, … (×1.5 per step; below 1× is 0); the toggled phases then move continuously, at 1× hue 1/8 turn per second and lightness 1/2. A step past 20× is refused (the top is 1.5⁷ ≈ 17.1×). The speed eases to each new value (time constant 0.15 s) |
| **`}` / `{`** (Shift+`]` / `[`) | the same by half-steps (×√1.5) |
| **`0`** | drift speed 0 (eases to a stop) |
| **↑ / ↓** | double / halve the maximum iterations (default 2048). Tiles are retargeted, not recomputed ([tiles](tiles.md#iterations-not-colours)). ↑ is **refused** when the doubled count would not fit the GPU's buffers (`MAX_ITERATIONS`, logged at start: 2²⁷ on an M2 Pro; the log warns, and the title shows `iters N (GPU max)` while at the ceiling); a pasted count above it is capped, with a warning in the log |
| **← / →** | sampling ratio s down / up by 0.5, within 0.5…4 (default 1) ([compositor](compositor.md#sampling-and-antialiasing)). On the web, lowered if the view would not fit the memory ceiling (shown `s 4 (→3)`) |
| **Space** | reset: drop every tile, the CPU reference lists, and the GPU backend's cached reference, failed-search memory and running search (`Perturbator::reset`), so it searches for a nucleus afresh |
| **Escape** | switch backend CPU ↔ GPU (and reset). The app **starts on the GPU** |
| **Cmd/Ctrl-C** | copy the view as `x,y|units_per_pixel/iterations` (`x,y` = the top-left corner, exact decimal) |
| **Cmd/Ctrl-V** | paste such a string: jump to that view (and iteration count) |

- Shift-scroll on a mouse wheel: macOS turns it into horizontal scrolling; the code takes the
  horizontal amount then, so Shift-zoom and Ctrl+Shift-scroll work.
- **Letter shortcuts match the character** the layout produces (`logical_key`), not the key
  position: the owner types Dvorak, so Cmd-C is the key labelled C on any layout. Space,
  Escape and the arrows use the physical key.
- **Cmd-scroll** for colour (the owner's call, 2026-11-04; Alt and Ctrl are toggles now). On
  Windows and Linux Cmd is the Windows/Super key, which opens the Start menu or the launcher
  when released: there colour scrolling is awkward, not yet addressed.
- The palette state (drift speed, toggled phases) is in the title: `drift +1.50× hue light⇅`.
- **On the web**, a view can be opened from the URL: `…/mandelbrot/#<Cmd-C string>` is
  applied like a paste.

## What the title shows

Diagnostics (marked `DIAG` in the code) in the window title (natively) or the debug line at
the top of the page (web): backend / pipeline (`cpu`, `f32`, `fe` = floatexp), view, depth,
upp (units per pixel), iterations, s (with `→` when lowered, `cache 1.4×` when the cache was
cut), palette phases, centre (f64: too coarse to navigate back with — use Cmd-C), held
modifiers, and the last key with what it did. If a thread panicked (the compute thread, say, which stops all
rendering), the title starts with `⚠ thread 'compute' crashed, see the log`
(`platform::crashed_thread`, recorded by a panic hook installed in `init_logging`).

## Logging

Natively, logs go to stderr **and** `gpu.log` in the working directory (overwritten each
launch, gitignored); the level defaults to `info`, `RUST_LOG` overrides it. On the web, to
the browser console. `[diag gpu]` lines give per-pass and per-chunk timings, references,
glitch rounds and per-chunk result statistics. `[diag tiles] gen G pass P done: … ms into
the generation, … ms since the first` times a whole render, across the generations that
restart it (e.g. after a resize): the "since the first" of the last pass.
