# Glossary

| term | meaning |
|---|---|
| **view** | units per screen pixel (a float); also loosely the visible region |
| **depth** | the tile grid's zoom level d; units per tile pixel `upp = 2^(−5−d)` |
| **upp** | units per (tile) pixel; `upp_log2(depth)` |
| **s** (sampling ratio) | tile pixels per screen pixel per axis, at least; 0.5…4, ←/→ ([compositor](compositor.md#sampling-and-antialiasing)) |
| **tile** | 128 × 128 pixels at one depth, the unit of computing and caching ([tiles](tiles.md)) |
| **group** | 16 × 16 tiles sharing one CPU reference list |
| **pass** | one of 7 coarse-to-fine sets of pixels in a tile (4 grid strides, 3 sub-passes) |
| **generation** | one render of a requested view: every pass over every visible tile (`run_generation`) |
| **chunk** (GPU backend) | a batch of a pass's pixels sent to the GPU in one go, ~30 ms of work |
| **chunk** (compositor) | a set of 2D array textures holding many tiles of one shape |
| **reference** | the one point whose orbit is computed exactly; every pixel iterates its difference from it ([perturbation](../explanation/perturbation.md)) |
| **nucleus** | the centre of a hyperbolic component (a minibrot or bulb), where z returns to 0 after p iterations; the GPU's preferred reference |
| **period p** | the length of a nucleus's cycle |
| **δ (delta)** | a pixel's offset from the reference orbit; δ₀ = the pixel's offset from the reference point |
| **glitch** | a pixel whose result cannot be trusted with this reference (the reference orbit ended before the pixel escaped) |
| **glitch round** | re-running glitched pixels against a better reference |
| **rebasing** | restarting a pixel against the start of the reference orbit when \|z\| < \|δ\| |
| **BLA** | bilinear approximation: skipping 2^k iterations at once while δ's step is linear |
| **interior detection** | declaring a pixel in the set early when its orbit contracts every period |
| **floatexp** | a float with an f32 mantissa and a separate i32 exponent, for depths beyond f32's exponent |
| **pipeline** | f32 or floatexp (`fe`): which shader a pass uses |
| **retarget** | adapting a tile to a new maximum iteration count without recomputing what is still valid |
| **seeding** | filling a tile's passes from its parent or children at the neighbouring depths |
| **Cmd-C string** | `x,y|units_per_pixel/iterations`, a view's exact position and iteration count |
