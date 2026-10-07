# Run the tests

What each test checks: [tests](../reference/testing.md).

1. **Always:** `cargo test` — no GPU needed, ~10 s once built. Baseline (2026-10-03): 35 passed,
   16 ignored, plus the benchmark files' 2 + 2 and the docs check.
2. **After touching the GPU backend, the shaders or the compositor:**
   `cargo test --release -- --ignored`. These run on this machine's GPU against exact `FBig`
   orbits; the 2⁻³⁰⁵ view's exact values take ~1 min unless cached: set `DIAG_OUT=<dir>`
   to cache them (and to dump renders as raw RGB).
3. **One test or measurement, with its output:**
   `cargo test --release -- --ignored diag_far_reference --nocapture`. The `diag_*` ones print
   numbers and assert nothing; some take parameters from the environment (`DIAG_TILES`,
   `DIAG_BASE`, `DIAG_SMOOTH` for `diag_recolour_speed`; `DIAG_LEVELS=-316,-320` and
   `DIAG_PIPE=1` for `diag_deep_nucleus_search`).
4. **Docs:** `cargo test --test docs`.

A GPU test that fails only after a change to struct layouts usually means a host/WGSL mismatch
([pitfalls](../reference/pitfalls.md#the-gpu)).
