# Backends and the compute thread

How a generation's passes reach a backend, and the CPU backend. The GPU backend has its own
page: [GPU backend](gpu-backend.md).

## The `Perturbator` trait (`src/tiles/perturb/mod.rs`)

```rust
pub trait Perturbator {
    fn render_pass_batch<'a>(&'a self, ctx: &'a PassBatchCtx, tiles: &'a [TileItem],
                             pass: u8, int: &'a MultiInterrupter) -> BatchFuture<'a>;
    fn reset(&self) {}   // Space: forget state kept across generations
}
```

- One call renders **one pass** for a batch of tiles, updating the tiles in place and calling
  `finish_pass` on each completed tile (a backend may skip that when interrupted).
- **It returns a future** (`BatchFuture`, not `Send`): on the web a worker cannot block on a
  GPU readback, which resolves from its event loop. Natively the compute thread drives it with
  `pollster`, and the GPU's `wait` still blocks on `device.poll`, so nothing changed there
  (checked: same timing).
- `PassBatchCtx`: the view's coordinates, the tile-grid depth and origin, the screen size, the
  iteration count, and the store's `progress` counter (bumped after tiles finish mid-pass so
  the compositor picks them up).
- `TileItem`: the tile, its group's CPU reference list (`refs`) and its offset in the group
  (`anchor_px`); the GPU ignores the last two.
- **`Toggle`** holds both backends and delegates by a shared `use_gpu: AtomicBool` (Escape
  flips it; it starts `true`).

## The compute thread (`src/drawing/renderer/multithreaded.rs`)

`compute_loop`: one async loop that makes the backends (its own `GpuState` — GPU objects never
leave their thread; on the web they are `!Send`) and awaits `run_generation`
([tiles](tiles.md#a-generation)) for each requested view, receiving views and interrupts over
a `waker_interrupter` channel. Natively a thread driven by `pollster`; on the web a Web Worker
([web build](web-build.md)).

## The CPU backend (`src/tiles/perturb/cpu.rs`)

The original algorithm, kept as the second backend:

- **f32 delta iteration** against reference orbits shared per **group** of 16 × 16 tiles
  (`RefList`, an append-only list, in `GroupCache` in `render.rs`).
- **A glitched pixel computes its own exact orbit** (`calculate_orbit`), and pushes it onto
  the front of its group's list so later pixels reuse it.
- Tiles run in parallel with rayon `par_iter`; `DETERMINISTIC = true` makes it sequential, in a
  fixed order, for reproducible output.
- **Black-fill:** if pass 0 and the tile's perimeter are all black (in the set), the whole tile
  is filled black and the remaining passes skipped. (The GPU deliberately has no such
  shortcut.)
- It lacks the GPU's nucleus references, rebasing, BLA and interior detection
  ([backlog](../backlog.md)).

`Pf` (the perturbation float, `src/rendering.rs`) is **f32** so the CPU matches the GPU's
precision for comparison tests; the CPU's natural setting is f64, which reaches far deeper.
**Ask the owner before changing it.**
