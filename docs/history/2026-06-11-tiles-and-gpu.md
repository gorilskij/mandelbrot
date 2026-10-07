# 2026-06-11 – 16: tiles, the GPU backend, winit

A rewrite over a few days, drawn from the commits.

- **06-11** — **Tree-based tiles**: a quadtree cache of tiles, progressive passes, and a
  compositor; the compositor picks the finest available source so resolution does not pop at
  depth boundaries. Reference orbits per tile, then per **group** of tiles behind one
  concurrent list (deterministic rendering, the long-orbit cost amortised). Reference-orbit
  precision fixed; exact f64 → `FBig` conversion everywhere (the lossy `to_fbig` deleted);
  cursor-locked zoom; a resizable window.
- **06-11** — The **`Perturbator` trait** extracted, and `Pf` (the perturbation float) switched
  to **f32** — then the first **wgpu GPU backend** (an "mvp" on 06-12).
- **06-13** — **winit** instead of the previous windowing (HiDPI, smooth pan and scroll), a
  lock-free tile map, and the **GPU compositor** ("amazing performance").
- **06-14** — CPU switched to f64 by default (reverted to f32 on 06-16: CPU and GPU must match
  for comparison testing), the progress bar, smooth window moving and panning, instant close,
  a zoom-out limit; old bug notes removed.
- **06-15 – 16** — The GPU renderer worked but showed artifacts past a certain zoom: WGSL moved
  out of string literals into files, and **floatexp** implemented; artifacts fixed. Branches
  `better-gpu-v1` and `blur` were left unmerged.

Next: [deep zoom on the GPU](2026-09-23-deep-zoom-on-the-gpu.md).
