# Repository layout

GitHub: `gorilskij/mandelbrot` (public). Default branch `master`.

```
CLAUDE.md                     rules for Claude and pointers into docs/
docs/                         this documentation
Cargo.toml, Cargo.lock        dependencies (Cargo.lock committed: the wasm-bindgen CLI must match)
rust-toolchain.toml           the pinned nightly, rust-src, the wasm target
.cargo/config.toml            the wasm target's flags (atomics, shared memory, TLS exports)
wrangler.toml                 the Cloudflare Workers (prod and [env.test])
bugs.txt                      views the owner found odd (Cmd-C strings), kept for reference
src/
  main.rs                     the app: window, input, view state, title/log diagnostics
  platform.rs                 everything that differs between native and web
  rendering.rs                calculate_orbit, check_orbit, check_divergence_delta, Pf, the palette (val_to_color, PalettePhase)
  gpu_compositor.rs           draws tiles to the screen (its own wgpu device)
  gpu_compositor_recolour.wgsl     colouring pass: palette, gap fill, mip levels
  gpu_compositor_downsample.wgsl   area-average downsample to the screen
  drawing/
    mod.rs                    the Drawer: owns the store and the compute thread
    maybe_pixel.rs            a tile pixel: iterations, or "not computed" (bit 31)
    renderer/multithreaded.rs the compute thread (compute_loop)
  tiles/
    store.rs                  grid math, passes, Tile, TileStore, the memory budget
    render.rs                 run_generation, GroupCache
    perturb/
      mod.rs                  the Perturbator trait, PassBatchCtx, TileItem, Toggle
      cpu.rs                  the CPU backend
      gpu.rs                  the GPU backend (and its test module: harnesses, GPU tests)
      firefox_nudge.rs        web only: the Firefox readback workaround (to remove when Firefox is fixed)
      nucleus.rs              nucleus search
      bla.rs                  the BLA table
      shaders/                floatexp.wgsl, perturb_common.wgsl, perturbation.wgsl, perturbation_floatexp.wgsl
  support/                    small utilities: an append-only list (append_only.rs), euclid-style points and scales
tests/
  docs.rs                     the docs check
  calculation_*_perf.rs, common/   early micro-benchmarks
web/                          the web build — see reference/web-build.md
```

Not in git: `target/`, `web/dist/`, `web/.tools/` (the wasm-bindgen CLI), `.wrangler/`,
`gpu.log`, `images/` and any `*.png` (local screenshots), `.idea/`, `flamegraph.svg`.
