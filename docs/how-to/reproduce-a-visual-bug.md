# Reproduce a visual bug

When the owner reports something that looks wrong at a view, **reproduce it in the test harness
first**, then A/B fixes there — rather than theorising, or trusting a CPU mirror of a shader.
Every deep fix of 2026-09 was found this way.

1. **Get the view.** The owner presses Cmd-C and pastes the string
   (`x,y|units_per_pixel/iterations`; `x,y` is the top-left corner). Note what is wrong and
   where (black where there should be colour, blocks, a wrong shape).
2. **Write an ignored test** in `gpu.rs`'s test module, named after the date
   (`view_YYYY_MM_DD…`), using the harness:
   - `sample_view(clip, iters, (w, h), (iw, ih))` — parse the string into a sampled view (a
     grid of sample pixels), with each sample's **exact** result computed from its own `FBig`
     orbit (in parallel; `v.exact`);
   - `gpu_on_sample(&v, &gpu)` — run the real shader on it (one reference, one pass;
     `gpu_on_sample_with(…, bla)` to switch BLA) — or `run_pipeline(&v, &backend, &store, …)`
     for whole generations (nucleus search, chunks, glitch rounds, resets);
   - `score(&got, &v.exact)` — (wrong, off by more than 50 iterations, black) counts.
   Look at the existing `view_2026_09_*` tests and copy one.
3. **Run it** (`cargo test --release -- --ignored <name> --nocapture`) and confirm the numbers
   show the bug (e.g. black counts vs exact). `DIAG_OUT=<dir>` dumps renders as raw RGB to
   look at.
4. **A/B the fix** in the test until it matches exact, then check the other GPU tests still
   pass and the regular suite is green.
5. **Record it:** the test stays (as a regression test, or a `diag_` measurement if it only
   prints), the numbers go in the commit message and the [history](../history/README.md), and
   the view goes in the reference page it concerns.

`bugs.txt` holds older odd-looking views (Cmd-C strings) kept for reference.
