# Firefox's late readbacks (2026-10-07)

**The report.** The owner found the web build very slow on the GPU in Zen (Firefox). The
`[diag gpu]` log showed every chunk at the 1024 px floor taking ~100 ms, for pixels that
escaped by iteration 51 (well under a millisecond of GPU work).

**The cause.** Firefox's GPU process polls its devices for finished work on a 100 ms timer,
or when the device gets a `queue.submit`
([bug 1870699](https://bugzilla.mozilla.org/show_bug.cgi?id=1870699), open; Mozilla's fix is
a polling thread that waits only while work is in flight, blocked on wgpu changes as of
2026-09-24). Our chunks are sized from submit → readback time, so the delay counted as cost
per pixel: with a fixed delay a, chunks settle at b·n = 30 ms − a, and a = 100 ms has no such
point, so they collapsed to the floor: ~10k px/s.

**How others handle it.** Submitting empty command buffers while waiting: measured in the bug
as 100 ms → 21 ms with one every 20 ms; Donner (a vector renderer) submits one once a readback
is late (raster 1.36 s → 0.31 s in Firefox); the RSX engine does the same. Mozilla asked that
such workarounds stay easy to find and remove.

**The decision** (with the owner): a nudge in one quarantined file, `firefox_nudge.rs`, web
only. `wait` waits until the chunk's estimated GPU time, then submits an empty command buffer
every 4 ms until the readback arrives.

**Measured** (M2 Pro, default view, all 7 passes, driven over WebDriver BiDi / CDP):

| | Firefox 157 | Chrome |
|---|---|---|
| `master` | pass 2 after 40 s (404k px; ~10k px/s) | 1.71, 1.74 s |
| nudge | 6.5, 11.4 s; ~1 nudge per chunk | 1.75, 1.80 s; ~0 nudges per chunk |

Chrome's ~3 % difference is within run-to-run noise. The first dispatch of a page (shader
compilation, estimated from the initial 1e-4 ms/px) gets up to ~130 nudges once.

**Rejected:**
- *Nudging from the moment of submit*: ~8 nudges per 30 ms chunk everywhere, for nothing in
  browsers that deliver on time.
- *A browser check (`Firefox/` in the user agent)*: targets the browser rather than the
  symptom, keeps nudging after Firefox is fixed, misses other browsers. Waiting for the
  estimated time already makes it fire only where readbacks are late.
- *A 10 ms interval*: a fixed delay a costs chunks 30 − a ms of work; 4 ms is also the
  browsers' clamp on nested timers, so shorter buys nothing.
- *Several chunks in flight*: unnecessary once readbacks are prompt, and two queued chunks
  break chunk timing ([GPU backend](../reference/gpu-backend.md)).
- *Raising the chunk floor on the web*: blunt, and lengthens interrupts.

**Found along the way:** the canvas swallows browser shortcuts and the context menu (winit's
`prevent_default`); Firefox shows the P3 canvas as sRGB (its WebGPU has no `colorSpace`).

**Correction, later that day: GPU timestamps.** With the nudge, submit → readback per pixel
was still 4–25× Chrome's, first read as Firefox computing slowly. Timestamps per dispatch
(`TIMESTAMP_QUERY`, both browsers have it) say otherwise; default view, two runs each:

| passes 2–6 | Firefox 157 | Chrome 154 |
|---|---|---|
| GPU work (timestamps) | 26–40 ns/px | 23–26 ns/px |
| submit → readback | 170–490 ns/px | 36–115 ns/px |
| `prep` (buffers, encoding) | 3–23 ms per pass | 2–12 ms per pass |

So the GPU is busy only ~10 % of a Firefox chunk. The cause is the nudge's own design: it
starts at an estimate taken from past submit → readback times, so a Firefox result (which
arrives only on a submit or the 100 ms timer) almost never comes earlier than estimated, and the
estimate never falls to the real cost. The earlier fixed-delay analysis assumed the delay was
counted from the GPU finishing, not from the estimate. Next step in the [backlog](../backlog.md).
Research found no report of Firefox computing slower per pixel; the one similar report
([wgpu#9199](https://github.com/gfx-rs/wgpu/issues/9199), "20× slower on Firefox") was the
100 ms poll again.
