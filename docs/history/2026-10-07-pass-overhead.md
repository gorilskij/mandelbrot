# The CPU work between passes (2026-10-07)

**Why.** With Firefox's readbacks fixed, the GPU still computed for only 15–24 % of a render
in every environment, native included. Temporary timers (reverted) showed where the compute
thread's time went, per render of the default view:

| ms | Firefox 157 | Chrome 154 | native |
|---|---|---|---|
| GPU compute (timestamps) | 330 | 328 | 305 |
| waiting on dispatches (submit → readback) | 916 | 632 | 503 |
| `log_tile_colors` (DIAG, after each pass) | 535 | 468 | 468 |
| `pass_dc` (a `hypot` per pixel) | 569 | 472 | 47 |
| collecting the pass's pixels | 136 | 108 | 241 |

The owner chose to take the two cheap ones first, one commit each, measured.

**How it is measured.** The default view, from the first generation's start to pass 6 done
(the new `[diag tiles] … pass P done` line; a resize can restart the generation natively),
3 runs each: native (a 3000×2000 window), and Firefox and Chrome driven over WebDriver BiDi /
CDP with fresh profiles. M2 Pro. GPU busy = the dispatches' timestamps over that time.

| ms to pass 6 (GPU busy) | native | Firefox 157 | Chrome 154 |
|---|---|---|---|
| baseline | 1396, 1337, 1367 (26–30 %) | 2360, 2552, 2455 (13–15 %) | 1789, 1832, 1835 (18–19 %) |
| no `log_tile_colors` | 1051, 881, 845 (39–45 %) | 2020, 2084, 1887 (17–19 %) | 1314, 1333, 1347 (25–26 %) |
| and `pass_dc` without `hypot` | 916, 827, 818 (36–38 %) | 1483, 1257, 1353 (24–28 %) | 922, 891, 928 (37–38 %) |

**`log_tile_colors` removed.** Added on 2026-09-23 with the GPU diagnostics, to tell whether a
"monochrome screen" came from the compute side, it hashed every pixel of every visible tile
after each pass (in the deployed build too). The per-chunk `[diag gpu]` lines already give the
same statistics (in-set, distinct values, the most common one), so it went entirely; the
per-pass line now logs only times.

**`pass_dc` without `hypot`.** The bound on |δ₀| for the BLA table's radii took one `hypot` per
pixel (~140 ms per pass in wasm, ~10 ms natively). It now takes the largest squared distance
and one `sqrt`: the same value (pixel offsets are integers below ~10⁴, so the squares are
exact in f64). All 16 GPU tests pass, the BLA ones included.

**Overall** (baseline → both): native ~1370 → ~850 ms, Firefox ~2450 → ~1360 ms, Chrome
~1820 → ~915 ms. What is left of the GPU's idle time: waiting on each dispatch beyond its
compute, and collecting each pass's pixels ([backlog](../backlog.md)).
