# The web build

The same code compiled to wasm on WebGPU, served at `gorilskij.com/mandelbrot/`. What differs
from native lives in `src/platform.rs` behind one interface; the build is in `web/`. Building
and serving: [build the web version](../how-to/build-the-web-version.md); deploying:
[deploy](../how-to/deploy.md).

## Requirements

- **Nightly Rust with `rust-src`**: std is rebuilt with atomics (`-Z build-std`). The toolchain
  is pinned in `rust-toolchain.toml` (`nightly-2026-09-25`, with the wasm target).
- **A browser with WebGPU** (Chrome, Edge, Arc, Safari 26+).
- **Cross-origin isolation** (for `SharedArrayBuffer`): the server sends
  `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.
  `web/index.html` checks `crossOriginIsolated` and `navigator.gpu` and says which is missing
  instead of starting.

## How it runs

- **Threads are Web Workers sharing one wasm memory** (`wasm_thread`, with our own worker
  script, `WORKER` / `worker_url` in `platform.rs`, because `wasm_thread`'s own uses
  wasm-bindgen's deprecated init call). It comes in two variants: a plain thread's worker
  closes when the thread function returns, like a thread; an async task's
  (`platform::spawn_async`, e.g. the compute loop) stays alive, since the function only starts
  a task on the worker's event loop. The flags in
  `.cargo/config.toml` enable atomics, bulk memory and mutable globals, shared imported memory
  up to 4 GiB, and the TLS exports wasm-bindgen's thread support needs (current nightlies no
  longer add the shared-memory link args for `+atomics`). `parking_lot` needs its `nightly`
  feature there, or parking panics.
- **The compute path is async** ([backends](backends.md#the-perturbator-trait-srctilesperturbmodrs)):
  a worker cannot block on a GPU readback. GPU objects never leave their thread (they are
  `!Send` on the web), so the compute loop makes its own `GpuState`.
- **The browser's main thread must never block** (a contended lock or a wait throws there):
  the compositor draws there (`WebRenderer`, on each animation frame) with no rayon
  (`platform::ui_map`); the compute loop, the nucleus search and rayon's pool run in workers.
  The main thread still takes the view channel's lock briefly
  (`waker_interrupter::Sender::send`): rare contention, but a real risk
  ([backlog](../backlog.md)).
- **The page** (`web/index.html`) is a full-window `<canvas id="canvas">` and a debug line
  (the native window title). The canvas size comes from the layout until winit's resize
  observer fires (`initial_size`). The surface is **Display P3**: natively the Metal layer has
  no colour space, so macOS shows the values unconverted (as P3 on a P3 display), and a
  browser's sRGB default looked duller.
- **Views in the URL:** `…/#<Cmd-C string>` opens that view, applied like a paste (for sharing
  views and reproducing them in a headless browser).
- **Memory:** wasm32 has 4 GiB; the tile store has a 3 GiB ceiling there, which can lower s
  ([tiles](tiles.md#the-memory-budget)).
- Logging goes to the browser console; clipboard copy/paste goes through the browser (which
  may ask for permission); some browser shortcuts win over the app's (Cmd-W; pinch or
  Ctrl-wheel zoom where the browser keeps the wheel event).

## The files

| file | what |
|---|---|
| `web/build.sh` | builds the wasm (`cargo build --release --target wasm32-unknown-unknown -Z build-std=std,panic_abort`), runs `wasm-bindgen` (installing the CLI version matching `Cargo.lock` into `web/.tools` if needed) and stages `web/dist/mandelbrot/` (`index.html`, `mandelbrot.js`, `mandelbrot_bg.wasm`) and `web/dist/_headers` |
| `web/serve.py` | serves `web/dist` at `http://localhost:8000/mandelbrot/` with the two headers and no caching |
| `web/cf-build.sh` | the Cloudflare build: installs rustup (the toolchain file brings the nightly, `rust-src` and the target), then `build.sh` with `CARGO_PROFILE_RELEASE_DEBUG=false` |
| `web/_headers` | the COOP/COEP headers for `/mandelbrot/*` (read by Workers static assets) |
| `web/worker.js` | the Worker's script: runs only when no file matches and passes the request on to the site (its 404 page) |
| `web/index.html` | the page |
| `wrangler.toml` | the Workers `mandelbrot` (prod) and `mandelbrot-test` ([deploy](../how-to/deploy.md)) |

- **`Cargo.lock` is committed**: the wasm-bindgen CLI must match the library version exactly.
- **The release profile keeps debug info** (for native profiling), which would make the wasm
  too big for static assets (25 MiB per file); `cf-build.sh` turns it off for the deployed build.
- `waker_interrupter` is a git dependency (the owner's own crate, public on GitHub).

## Firefox reads back late

Firefox's GPU process (as of Firefox 155, 2026-10) learns that GPU work has finished only
from a timer that polls every 100 ms, or when the same device gets a `queue.submit`. Every
`mapAsync` and `onSubmittedWorkDone` therefore resolves up to 100 ms late, however little the
work computes: measured in Zen on 2026-10-07 as ~100 ms per 1024 px chunk of cheap pixels.
Submitting an empty command buffer while waiting delivers the result at the next submit
(another site measured 100 ms → 21 ms with one every 20 ms). Submits on another device (the
compositor's) do not help. Mozilla's fix (a polling thread that waits only while work is in
flight) is
[bug 1870699](https://bugzilla.mozilla.org/show_bug.cgi?id=1870699), open as of 2026-09-24.
Chunk sizing does not account for this yet ([backlog](../backlog.md)).

## Checked so far

In headless Chrome driven over CDP: rendering, zoom, palette, s, the memory ceiling, loading
from `/mandelbrot/` with cross-origin isolation (2026-10-02). **Deep zoom** (the floatexp
pipeline, BLA, the nucleus search at depth) has **not** been checked in a browser.
