# Build and serve the web version

How it works: [web build](../reference/web-build.md).

```
web/build.sh       # → web/dist/mandelbrot/ (+ web/dist/_headers)
web/serve.py       # http://localhost:8000/mandelbrot/   (web/serve.py 9000 for another port)
```

- The first run installs the wasm-bindgen CLI matching `Cargo.lock` into `web/.tools/`
  (`cargo install wasm-bindgen-cli --version <locked> --locked`); a version bump of
  `wasm-bindgen` in `Cargo.lock` makes it reinstall.
- The build is a release build **with debug info** (the release profile keeps it for native
  profiling) — fine locally; `bash web/cf-build.sh` builds what is deployed (no debug info).
- `serve.py` sends the COOP/COEP headers and `Cache-Control: no-store`, so a rebuild shows on
  reload.
- Open it in a WebGPU browser. If the page says "Not cross-origin isolated", the headers are
  missing (another server?); "WebGPU is not available", the browser lacks it.
- Logs are in the browser console. A view: append `#<Cmd-C string>` to the URL.

To check it like production (the Worker and its `_headers`), run `npx wrangler dev` after a build
and open `http://localhost:8787/mandelbrot/` (misses there loop locally: the site's Worker,
which answers them in production, is not running).
