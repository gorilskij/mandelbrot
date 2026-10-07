# 2025-01 – 2026-03: from a CPU renderer to perturbation

Drawn from the commits (short messages; reasons recorded where known).

## 2025-01 – 02: the first renderer

- **01-11** — A multithreaded CPU renderer: escape-time colouring with HSL, symmetry (tried,
  removed the same day), then interactive zooming.
- **01-15 – 29** — Zoom in and out, background rendering (lock contention fixed by switching to
  `parking_lot`), a missed-update bug when zooming stopped early, dragging.
- **02-01 – 09** — Benchmarks (the files still in `tests/`), a separate draw thread, rendering
  in parallel boxes radiating from the centre, a variable iteration count, a consistent colour
  scheme. Branches `f128` (128-bit floats) and `ill-fated` (a draw thread) were left unmerged.

## 2025-10 – 11: deep zoom on the CPU

- **10-15** — Rendering centred on the cursor when possible; no buffer updates while
  stationary. `simd` ("BROKEN - WIP") and `change-z` (arrows to change z₀) left unmerged.
- **11-02** — The **black-square optimisation** (a block whose outline is in the set is filled
  without computing it — the ancestor of today's CPU black-fill).
- **11-03** — **Arbitrary-precision bigfloats** (`dashu`) and **deltas**: perturbation against a
  reference orbit (branches `deltas` / `implement-deltas` / `delta_performance`;
  `scaled_precision`, "this was not to be", left unmerged).
- **11-12 – 16** — **Copy/paste of coordinates** (with the iteration count); a **list of
  reference orbits** that later pixels reuse (prepended, so the newest is tried first).
- **11-19 – 20** — Continuous generation (rendering while zooming), partial buffers.

## 2026-03: render order

- **03-04** — An interpolation-friendly chunk rendering order.

Next: [tiles and the GPU](2026-06-11-tiles-and-gpu.md).
