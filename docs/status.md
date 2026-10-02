# Status

The living "where are we" page. Update it whenever any of this changes. Dates are absolute.
Last updated: **2026-10-03**.

## In one paragraph

The explorer works natively (macOS, Metal) and on the web (WebGPU), with the GPU backend as
default: nucleus references found in the background, rebasing, BLA, interior detection, a
floatexp pipeline below 2⁻¹⁰⁰, chunked interruptible dispatch, and a compositor that colours
on the GPU with exact-area antialiasing at an adjustable sampling ratio. Views around 2⁻³²⁰
render in seconds. The web build is live at `gorilskij.com/mandelbrot/` (since 2026-10-02, as
a Cloudflare Worker; before that `games.gorilskij.com/mandelbrot/` on Pages). Open threads:
[backlog](backlog.md).

## Deployed

| | where | from | as of |
|---|---|---|---|
| prod | `gorilskij.com/mandelbrot/` | `pub-website` | 2026-10-02 (the code of 2026-09-25, 3833930, plus the hosting change) |
| test | `test.gorilskij.com/mandelbrot/` (behind Access) | `test-website` | the same |

`master` has one more docs-only commit and these docs; no code differs from what is deployed.

## Branches

| branch | state (2026-10-03) |
|---|---|
| `master` | the working branch and GitHub's default; everything merged |
| `test-website`, `pub-website` | deploy branches ([deploy](how-to/deploy.md)) |
| `bla`, `fix-glitches`, `gpu-recolour`, `interior-fix`, `palette-scroll`, `s-ceiling`, `wasm`, `better-gpu-v2`, `colors`, `continuous-generation`, `delta_performance`, `implement-deltas`, `performance`, `ref_orbit_list_prepend` | merged feature branches, kept as pointers |
| `deltas` (13 commits), `scaled_precision` (7), `f128` (4), `blur` (2), `better-gpu-v1`, `change-z`, `ill-fated`, `ref_orbit_list`, `simd` (1 each) | **unmerged** experiments, 2025-01 – 2026-06 (e.g. `simd` "BROKEN - WIP", `scaled_precision` "this was not to be") |
| `workers` | the 2026-10-02 hosting change, merged; local only |

## Tests

`cargo test` (2026-10-03, M2 Pro): **35 passed, 16 ignored** (the GPU tests) in the crate, 2 + 2
in the benchmark files, and the docs check. `cargo test --release -- --ignored` (the GPU
tests and measurements): **16 passed**, 3.5 min (2026-10-03).

## Uncommitted work

None (2026-10-03).
