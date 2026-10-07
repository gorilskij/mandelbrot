# Controls

Input handling is in `src/main.rs` (`App::window_event`). The same controls work natively
and on the web.

| input | does |
|---|---|
| scroll | zoom around the cursor: each wheel notch (a trackpad's ~10 px) scales the view by 2.5 %, at most 20 % per event (`apply_scroll_zoom`). The zoomed view shows at once, interpolated from existing tiles, and is recomputed around the cursor. Zooming out and panning stop at the start-up view's rectangle (`bounds`, `clamp_to_bounds`) |
| drag (left button) | pan |
| **Alt**-scroll | shift the palette's **hue** phase, 1/128 turn per notch (recolours only, nothing recomputed) |
| **Ctrl**-scroll | shift the palette's **lightness** phase, 1/32 turn per notch; **Shift** inverts the direction |
| Alt+Ctrl-scroll | both |
| **↑ / ↓** | double / halve the maximum iterations (default 2048). Tiles are retargeted, not recomputed ([tiles](tiles.md#iterations-not-colours)) |
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
- Alt rather than Cmd for the hue, so it works on every platform (the Windows key opens the
  Start menu); the owner unbound their own Alt binding for it.
- **On the web**, a view can be opened from the URL: `…/mandelbrot/#<Cmd-C string>` is
  applied like a paste.

## What the title shows

Diagnostics (marked `DIAG` in the code) in the window title (natively) or the debug line at
the top of the page (web): backend / pipeline (`cpu`, `f32`, `fe` = floatexp), view, depth,
upp (units per pixel), iterations, s (with `→` when lowered, `cache 1.4×` when the cache was
cut), palette phases, centre (f64: too coarse to navigate back with — use Cmd-C), held
modifiers, and the last key with what it did.

## Logging

Natively, logs go to stderr **and** `gpu.log` in the working directory (overwritten each
launch, gitignored); the level defaults to `info`, `RUST_LOG` overrides it. On the web, to
the browser console. `[diag gpu]` / `[diag tiles]` lines give per-pass and per-chunk timings,
references, glitch rounds and result statistics. `[diag tiles] gen G pass P done: … ms into
the generation, … ms since the first` times a whole render, across the generations that
restart it (e.g. after a resize): the "since the first" of the last pass.
