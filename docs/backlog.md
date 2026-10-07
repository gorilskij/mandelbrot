# Backlog

Everything open, in one place. Remove an item when it is done (and record it in the
[history](history/README.md)); add one the moment it is found. Items marked **(owner)** wait on
the owner's decision. **Talk an item through with the owner before coding it.** Last reviewed
**2026-10-03** (the open threads as of 2026-09-26, carried over).

## Open threads

1. **Share one wgpu device** between the compositor and compute. (The CPU/GPU overlap half of
   this item is done.)
2. **The offscreen image's size** (note only; the owner doesn't care for now): it is always 2×
   the surface per axis but fully used only just before a depth switch; sizing it to the ratio
   in use would save ~1 ms of GPU bandwidth per frame. It is scratch space, so no extra
   computation.
3. **Mantissa precision**: double-single delta arithmetic, or accept f32 (much better with
   BLA) ([precision](reference/precision.md#the-remaining-limit-the-mantissa)). **(owner)**:
   undecided.
4. **The CPU backend** lacks the GPU's nucleus references, rebasing, BLA and interior
   detection; the GPU lacks the CPU's black-fill. Not solid guessing (filling a cell whose
   corners match): the owner said not yet.
5. **High s**: still to measure whether compute chunks shrink to the minimum at s = 4 (its
   slowness is mostly the 16–64× work). Not done: freeing mostly-empty compositor chunks (a
   chunk is dropped only when entirely empty, so eviction can leave sparse ones).
6. **Don't wait for the search at the end of pass 0** (optional, only if the ~1–2 s wait still
   bothers the owner): move on to later passes while deferred pixels are pending. Tricky: a
   tile finishes a pass only once all its pixels are stored, and later passes would defer
   more pixels too.
7. **The main-thread lock in the view channel (web)** — TODO (owner).
   `waker_interrupter::Sender::send` (called on the browser's main thread on every view
   change) takes the channel's `parking_lot` mutex, which the compute worker also holds
   briefly (every `interrupted()` check between chunks, and while waiting in
   `recv_multithreaded`). If contended, parking calls `Atomics.wait`, which throws on the main
   thread; `notify_one` can too (parking_lot's internal bucket lock). Rare (nanosecond holds)
   but possible, e.g. zooming during a render. Fix in `waker_interrupter`: a lock-free send
   (store the message in an atomic slot, wake the receiver without blocking) and an
   atomic-flag interrupter check.

8. **The GPU idles most of a render**, measured 2026-10-07 (default view, temporary timers,
   reverted): GPU compute is 15–24 % of the time the compute thread spends. Per render:

   | ms | Firefox 157 | Chrome 154 | native |
   |---|---|---|---|
   | GPU compute (timestamps) | 330 | 328 | 305 |
   | waiting on dispatches (submit → readback) | 916 | 632 | 503 |
   | `log_tile_colors` (DIAG, after each pass, hashes every pixel) | 535 | 468 | 468 |
   | `pass_dc` (a `hypot` per pixel, for the BLA radius bound) | 569 | 472 | 47 |
   | collecting the pass's pixels | 136 | 108 | 241 |

   (Native: a 3000×2000 window, 18.4M px against 14.2M, and one restarted generation.)
   Candidates, to talk through: make `log_tile_colors` cheap or skip it unless asked for (it
   is diagnostics); `pass_dc` from squared distances (one `sqrt`) or the pixels' bounding
   box; then the rest of each dispatch's wait (four buffers created per dispatch, the
   `NOT_RUN` fill uploaded, a new staging buffer; in Firefox also the nudge's granularity), and
   the CPU work between passes, during which the GPU has nothing queued.
9. **The page swallows browser shortcuts and the context menu (web)**, reported 2026-10-07: in
   Firefox/Zen, right-click does nothing and Cmd-L (and likely other browser shortcuts) don't
   work over the canvas. Probably winit's web default `prevent_default = true` (canvas set up
   in `src/main.rs`, `with_canvas`), which cancels every keyboard, wheel and context-menu
   event. Possible fix: `with_prevent_default(false)` and cancel only the events the app uses.

10. **Colours differ in Firefox (web)** — TODO (owner, 2026-10-07): convert P3 → sRGB where unsupported. Reported 2026-10-07: Firefox's WebGPU canvas has no
    `colorSpace`, so our Display P3 values are shown as sRGB: duller
    ([web build](reference/web-build.md#how-it-runs)). Firefox has no wide gamut anywhere
    yet: on macOS it tags its windows sRGB by default (`gfx.color_management.native_srgb`),
    so even CSS `color(display-p3 …)` is clipped to sRGB; Mozilla's wide-gamut work
    ([bug 1626624](https://bugzilla.mozilla.org/show_bug.cgi?id=1626624)) has no date. The
    usual practice (three.js, for one): colour in a working space, convert to the output space
    in the final shader, and request `display-p3` only where `getConfiguration()` reports it
    (some add `matchMedia("(color-gamut: p3)")`). For us: detect the missing support and
    convert P3 → sRGB in the compositor's shader (linearise, 3×3 matrix, clip, re-encode):
    the same colours as Chrome wherever sRGB can show them. Other canvases in Firefox 157:
    2D ignores `colorSpace: "display-p3"`; WebGL2 accepts `drawingBufferColorSpace =
    "display-p3"` (unchecked whether it shows P3; moot while the window is sRGB).

## Maintenance

- **Move the pinned nightly forward** now and then ([update the toolchain](how-to/update-the-toolchain.md));
  `+atomics` is being phased out, so a newer one may need changes to the web build.
- **Remove the Firefox readback workaround** (`firefox_nudge.rs`) once Firefox ships
  [bug 1870699](https://bugzilla.mozilla.org/show_bug.cgi?id=1870699)
  ([pitfalls](reference/pitfalls.md#the-web)).
- **Publish `waker_interrupter`** (e.g. on crates.io) and depend on a version instead of the
  git repo (`Cargo.toml` TODO).
- **Check deep zoom in a browser**: the floatexp pipeline, BLA and the nucleus search at depth
  have only been tested natively ([web build](reference/web-build.md#checked-so-far)).
- **The two benchmark files in `tests/`** (2025-02) measure plain f64 loops that no current
  code uses. **(owner)**: keep or delete.
- **`bugs.txt`** lists odd-looking views from before the 2026-09 GPU work; whether each still
  reproduces has not been re-checked.
- **Old branches** (`deltas`, `f128`, `scaled_precision`, `simd`, `change-z`, `blur`,
  `better-gpu-v1`, `ill-fated`, `ref_orbit_list`) hold unmerged experiments, most from 2025
  ([status](status.md#branches)). **(owner)**: keep or delete.

## Decided (do not re-propose)

- **Render order stays pass-major** (each pass completes over the view before the next starts);
  ring-by-ring refinement was rejected by the owner (2026-09-24).
- **No |δ|/|X| glitch heuristic** ([perturbation](explanation/perturbation.md#perturbation)).
- **No memory ceiling natively** (the owner wants s = 4 unrestricted); only on the web.
- **Iterations stay on the GPU** for fast recolouring (~1.5 GiB at s = 4), accepted by the owner.
