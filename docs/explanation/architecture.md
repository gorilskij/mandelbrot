# Architecture

## The shape

```
 main thread (winit)            compute thread / worker            render thread (native)
 ┌──────────────────────┐      ┌────────────────────────────┐      or main thread (web)
 │ input, view state,   │ view │ compute_loop               │      ┌──────────────────────┐
 │ title diagnostics    │─────▶│  run_generation per view   │      │ GpuCompositor        │
 │ (main.rs)            │ (+   │   for each pass:           │      │  upload changed tiles│
 └──────────────────────┘ int.)│    backend.render_pass_    │      │  colour (palette)    │
                               │    batch (CPU or GPU)      │      │  draw + downsample   │
                               └─────────────┬──────────────┘      └──────────▲───────────┘
                                             │ writes                         │ reads
                                             ▼                                │
                                   ┌──────────────────────────────────────────┴──┐
                                   │ TileStore: tiles of escape iterations,       │
                                   │ quadtree of depths, LRU budget               │
                                   └──────────────────────────────────────────────┘
```

- **The view is the only input to computing.** Input changes coordinates and publishes the view
  over a `waker_interrupter` channel; the compute thread abandons the current generation
  (checked between chunks) and starts the new one.
- **Tiles hold iterations, not colours**, so recolouring (palette scrolling) never recomputes,
  and changing the iteration limit mostly doesn't either ([tiles](../reference/tiles.md)).
- **Drawing and computing are decoupled through the store.** The compositor draws whatever is
  there — parents and children at other depths fill in while the current depth computes, so
  zooming shows something at once.
- **Two compute backends behind one trait** (`Perturbator`): the GPU one (default) and the
  older CPU one, switchable with Escape for comparison ([backends](../reference/backends.md)).
- **The compositor has its own wgpu device**, separate from the GPU backend's (sharing one is
  an open thread).

## Why these choices

- **Tiles on a quadtree** (2026-06): caching across zoom and pan, and free coarse previews
  (a child's first passes are its parent's pixels).
- **Progressive passes, pass-major order**: a usable picture of the whole view first, then
  refinement. Ring-by-ring refinement (finish near the cursor first) was rejected by the owner
  (2026-09-24).
- **The GPU as the main backend** (2026-06, default since 2026-09-24): orders of magnitude
  faster per pixel; the deep-zoom work of 2026-09 (nucleus references, rebasing, BLA, interior
  detection, floatexp) went there.
- **Chunks of ~30 ms** on the GPU: interruptible within a frame or two, and short enough that
  macOS does not kill the command buffer.
- **The web build shares all the code** (2026-09-25): the compute path became async rather
  than forking the code; `platform.rs` holds the differences.

The math: [perturbation](perturbation.md). Dated decisions: [history](../history/README.md).
