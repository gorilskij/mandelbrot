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

8. **The GPU still idles over half a render** (2026-10-07, after removing `log_tile_colors`
   and the per-pixel `hypot`: [history](history/2026-10-07-pass-overhead.md)): busy 36–38 %
   natively and in Chrome, 24–28 % in Firefox. Left, to talk through: the wait on each
   dispatch beyond its compute (submit → readback was 503 / 632 / 916 ms per render natively /
   Chrome / Firefox against ~320 of compute: four buffers created per dispatch, the `NOT_RUN`
   fill uploaded, a new staging buffer; in Firefox also the nudge's granularity and its
   process boundary); collecting each pass's pixels (108–241 ms per render); the CPU work
   between chunks and passes, during which the GPU has nothing queued.
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

12. **The compute thread freezes on high iteration counts at depth (web)**, reported and
    reproduced 2026-11-04 (the owner's view
    `-1.962806555174172021647685634946390363305838735256282905291651714974987176375827993719362540520237766236256620207342904928884631910114733336316139198285994494676342,-0.000000000000000000000000000000000000000000000000000000000000000042947767643305669232409999573568561541838549727767369923819385466754528090057229022273635574237949421053083159607872189092473578715032590666576834883379438641201333430314048627108223712715224994612480066102195044615076323678|2.346020205021077e-134/4096`,
    2⁻⁴⁴⁴, then ↑ to 1M): in Chrome the view stayed frozen ~39 s after the last ↑ (progress
    bar stuck; ↓, zoom and pan wait too). Natively the same sequence finishes, 1M in 7.5 s.
    Cause: at each new iteration count `initial_reference` computes the view centre's orbit
    (`Reference::new` → `calculate_orbit`, bigfloat) on the compute thread, and that computation
    checks no interrupt: 17.5 s at 1M in wasm (2.3 s natively). Then, interrupted by a newer
    message, it **discarded** the finished orbit (`if int.interrupted() { return None; }`) and
    the next generation computed it again (11.9 s); the nucleus search took 28 s meanwhile (in
    the background, but pass 0 waits for it). To talk through: make `calculate_orbit`
    interruptible (check every ~1000 iterations); keep a finished orbit instead of dropping it;
    and later extend the centre's orbit on ↑ instead of recomputing it from scratch.

    **Natively the freeze is a crash** (reproduced 2026-11-04 by driving the native app with
    System Events, the owner taking it on to 8M and zooming): the compute thread panicked on a
    fatal wgpu validation error, `Device::create_buffer: Buffer size 536870880 is greater than
    the maximum buffer size (268435456)`, i.e. the BLA table of an 8M-iteration orbit
    ((2²⁴ − 1) × 32 bytes). With the compute thread dead nothing renders again, whatever the
    iterations. The device is created with wgpu's default limits (256 MiB per buffer, 128 MiB
    per storage binding); the M2 Pro's adapter allows 4 GiB for both. Sizes by iterations:
    BLA table ≈ 2 × 2^⌈log2 n⌉ × 32 B (level 0, unused by the shaders, is half of it), the
    floatexp orbit n × 16 B; so 2M is the last count that fits the defaults (table 134217696 B
    of a 134217728 binding). To talk through: request the adapter's limits; check sizes against
    `device.limits()` before creating buffers and degrade (cap the BLA table, refuse or cap the
    iterations, visibly) instead of panicking; drop the unused level 0 from the table; and make
    a dead compute thread visible instead of a silent freeze.

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
