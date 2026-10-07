# Tests

Baseline (2026-10-03, M2 Pro): `cargo test` — **35 passed, 16 ignored** in the main crate, plus
2 + 2 in the two benchmark files; `cargo test --test docs` — the docs check;
`cargo test --release -- --ignored` — **16 passed** in 3.5 min. Running them:
[run the tests](../how-to/run-the-tests.md).

## Regular tests (no GPU needed)

In the modules they test (`#[cfg(test)]`): pass layout and `pass_pixels`/`pass_of`, seeding
geometry, retargeting, the tile budget and `effective_ratio` (`store.rs`), the compositor's
level choice (`drawn_levels_cover_the_ratio_range`) and progress bar (`bar_*`), nucleus search
(`period_2_nucleus`, `airplane_and_rabbit`, `nucleus_reference_matches_exact_orbits`, …),
the BLA table (including "a jump equals stepping"), the CPU mirror of the reference path,
interior detection's safety (`interior_detection_is_safe_and_useful`),
`entry_layout_matches_wgsl` (the `BlaEntry` / WGSL `Bla` layout), and `shaders_validate`
(parses and validates every WGSL source with naga, since it is otherwise only compiled when the
GPU backend starts).

## GPU tests and measurements (`#[ignore]`)

`cargo test --release -- --ignored` runs them on this machine's GPU against exact `FBig`
orbits, at views the owner reported:

| test | checks |
|---|---|
| `artifact_view_2026_09_23_matches_exact` | 2⁻²²⁰ |
| `view_2026_09_23b_pipeline_matches_exact` | 2⁻³⁰⁵, the whole pipeline including resets |
| `view_2026_09_23c_matches_exact` | 2⁻³¹⁴, 262 144 iterations (the floatexp orbit) |
| `dispatch_is_deterministic_even_when_killed` | dropped dispatches are retried, results identical |
| `bla_with_escaping_reference_matches_plain` | BLA near an escaping reference |
| `zero_delta_escapes_with_reference` | a pixel at its own reference escapes with it |
| `recolour_matches_cpu` | the compositor's colouring vs `val_to_color` |
| `view_2026_09_25_black_disc` | a nucleus of another period: prints black counts vs exact, no assertion |

and `diag_*` measurements (print, no assertion):

| measurement | measures |
|---|---|
| `diag_recolour_speed` | a full-screen recolour (`DIAG_TILES`, `DIAG_BASE`, `DIAG_SMOOTH`) |
| `diag_bla_ab` | BLA on/off |
| `diag_switch_threshold` | the floatexp → f32 switch exponent |
| `diag_interior_detection` | interior detection's cost and false positives |
| `diag_view_2026_09_25b_references` | nuclei of other periods vs the view's own, with and without interior detection |
| `diag_reference_precision` | f32 rounding of long-lived pixels |
| `diag_far_reference` | accuracy with a reused far-away nucleus |
| `diag_deep_nucleus_search` | search timing below the 2⁻³¹⁴ view (`DIAG_LEVELS=-316,-320`, `DIAG_PIPE=1` for whole generations) |

`DIAG_OUT=dir` caches exact values (the 2⁻³⁰⁵ ones take ~1 min) and dumps renders as raw RGB.

## Harness for visual bugs

In `gpu.rs`'s test module: `sample_view` parses a Cmd-C string into a sampled view,
`gpu_on_sample` runs the real shader on it, `run_pipeline` runs whole generations,
`score` compares against exact orbits. See [reproduce a visual bug](../how-to/reproduce-a-visual-bug.md).

## Benchmarks (`tests/`)

`tests/calculation_loop_perf.rs` and `tests/calculation_exit_condition_perf.rs` (with
`tests/common/`) are early (2025-02) micro-benchmarks of plain f64 escape loops
(`#![feature(test)]`, nightly). They predate perturbation and test no current code; `cargo
test` runs their two tests each.

`tests/docs.rs` is the docs check.
