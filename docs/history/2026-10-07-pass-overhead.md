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

**Does the CPU work scale with the GPU's?** (2026-10-07, browsers, one run each.) Three views:
the default (2048 iterations), `bugs.txt`'s "black is not black" (1.05e-54, 32768 iterations,
floatexp) and the GPU tests' view "b" (1.8e-92, 2⁻³⁰⁵, 32768 iterations). Opened from the URL
hash, so generation 1 is the view itself.

| | default | 1e-54 | 2⁻³⁰⁵ |
|---|---|---|---|
| total, Chrome / Firefox | 922 / 1483 ms | 5305 / 6220 ms | 2824 / 3316 ms |
| pass 0, Chrome / Firefox (GPU busy) | 47 / – ms | 2341 / 2448 ms (2 %) | 1833 / 1918 ms (1 %) |
| GPU busy in passes 3–6, Chrome / Firefox | 36–40 % / 24–28 % | 83–85 % / 73–77 % | 70–76 % / 60–63 % |

The per-pixel CPU work does not grow with the iterations, so harder pixels raise the GPU's
share of passes 1–6. What grows is pass 0, nearly all CPU, before and while the first pixels
are computed: on the 1e-54 view in Chrome, 1.6 s in `run_generation` building the batch before
the backend starts, then a provisional reference orbit (356 ms) and the wait for the nucleus
search (325 ms). The 1.6 s is `group_list`: a bigfloat reference orbit per group of tiles,
sequential, which only the CPU backend uses (`TileItem::refs`); the GPU backend never reads
them ([backlog](../backlog.md)).

**The group orbits moved into the CPU backend** (the owner: "that needs to change"). Apart from
them the CPU backend was already dormant while the GPU ran (`Cpu` was a stateless unit
struct). Now `Cpu` owns the `GroupCache` and builds a pass's group lists itself; `TileItem` is
just the tile, and `run_generation` no longer computes anything per group. The moved code is
unchanged. Checked by hand: Escape to the CPU backend renders all 7 passes (Firefox; its pass
0 now includes the group orbits, 1.3 s at the default view). Browsers, two runs each after:

| | before | after |
|---|---|---|
| 1e-54: total, Chrome / Firefox | 5305 / 6220 ms | 4477, 4784 / 5085, 5322 ms |
| 1e-54: pass 0, Chrome / Firefox | 2341 / 2448 ms | 1353, 1448 / 1443, 1364 ms |
| 2⁻³⁰⁵: total, Chrome / Firefox | 2824 / 3316 ms | 1839, 1942 / 2415, 2259 ms |
| 2⁻³⁰⁵: pass 0, Chrome / Firefox | 1833 / 1918 ms | 869, 944 / 869, 898 ms |

The default view, 3 runs each: native 829, 843, 820 ms; Firefox 1294, 1339, 1356; Chrome 895,
883, 891 (before: 916, 827, 818 / 1483, 1257, 1353 / 922, 891, 928): unchanged, one group
orbit of 2048 iterations having been cheap. Native hard views were not measured (opening a view
natively needs a paste). What is left of pass 0 on the hard views: the provisional reference
orbit and the wait for the nucleus search, which now overlap (678 and 617 ms on the 1e-54 view).
