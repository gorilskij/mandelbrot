# Web build

The same app, compiled to wasm and running on WebGPU. Threads are Web
Workers sharing one wasm memory: the compute loop runs in a worker (async,
since a worker can't block on GPU readbacks), rayon's pool and the nucleus
search in others; the compositor draws on the page's main thread on each
animation frame.

## Requirements

- nightly Rust with `rust-src` (std is rebuilt with atomics: `-Z build-std`)
- a browser with WebGPU (Chrome, Edge, Arc, Safari 26+)
- the page must be *cross-origin isolated* (for `SharedArrayBuffer`): the
  server sends `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`

## Build and run locally

    web/build.sh          # -> web/dist (installs the matching wasm-bindgen CLI into web/.tools once)
    web/serve.py          # http://localhost:8000, with the headers above

## Deploy (Cloudflare Pages)

Upload `web/dist` (e.g. `npx wrangler pages deploy web/dist`); the `_headers`
file in it sets the two headers.

## Differences from native

- Logging goes to the browser console (no `gpu.log`); the debug line is an
  overlay at the top of the page instead of the window title.
- Memory: wasm32 has 4 GiB, so the tile store has a 3 GiB ceiling and a high
  s is lowered for big windows (shown as `s 4 (→3)`; see CLAUDE.md, tile
  budget).
- Clipboard (Cmd-C / Cmd-V) goes through the browser, which may ask for
  permission.
- Some browser shortcuts win over the app's (e.g. Cmd-W, and pinch/Ctrl-wheel
  zoom where the browser doesn't hand the wheel event to the page).
