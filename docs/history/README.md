# History

What was done, when, and **why** — including what was tried and rejected. The reference pages
say what is true now; this says how it came to be. Newest last.

| dates | entry |
|---|---|
| 2025-01 – 2026-03 | [From a CPU renderer to perturbation](2025-01-11-first-versions.md) — the first renderer, bigfloats and deltas, reference-orbit lists, copy/paste |
| 2026-06-11 – 16 | [Tiles, the GPU backend, winit](2026-06-11-tiles-and-gpu.md) — the quadtree of tiles, `Perturbator`, wgpu, the compositor, floatexp |
| 2026-09-23 – 24 | [Deep zoom on the GPU](2026-09-23-deep-zoom-on-the-gpu.md) — nucleus references, chunked dispatch, rebasing, interior detection, BLA, the floatexp orbit, background search, s and antialiasing |
| 2026-09-25 – 26 | [Palette scrolling, GPU colouring, the web build](2026-09-25-colouring-and-the-web.md) — chunks of array textures, the wasm memory ceiling, WebGPU and Web Workers, the interior return test |
| 2026-10-02 – 03 | [Hosting on Workers, and these docs](2026-10-02-workers-and-docs.md) — `gorilskij.com/mandelbrot/`; `docs/` |

The notes these entries were drawn from are kept verbatim:
[working notes as of 2026-10-03](working-notes-2026-10-03.md).

## Writing an entry

- One file per piece of work, named `YYYY-MM-DD-topic.md`; add it to the table.
- Absolute dates. Who decided what ("the owner's call") when it matters.
- The decision, **its reasons, the measurements, and the alternatives rejected** — a rejected
  idea recorded stops it being proposed again.
- Keep reference pages free of history: update them to the new truth and link here.
