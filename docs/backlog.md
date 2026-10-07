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

8. **Slow on Firefox (web)**, reported 2026-10-07. Measured in Zen (Firefox) that day: chunks
   stuck at the 1024 px floor, each taking ~100 ms from submit to readback, for cheap pixels
   (escaped by iteration 51, so the computation itself takes about a millisecond). Cause:
   Firefox's GPU process polls for finished GPU work on a 100 ms timer, and the other trigger is
   a `queue.submit` on the same device
   ([web build](reference/web-build.md#firefox-reads-back-late)). With one chunk in flight,
   `record_chunk` counts that delay as cost per pixel, so `chunk_len` shrinks to `MIN_CHUNK_PX`
   and the GPU mostly idles: about 10k px/s. Candidate fixes, to talk through:
   - submit an empty command buffer every few ms while a readback is pending (the known
     workaround; removable once Firefox fixes its bug);
   - fit time = a + b·n and size chunks from b only;
   - keep several chunks in flight (each submit then also delivers earlier results).
   Agreed 2026-10-07: the workaround stays quarantined (one marked place, easy to strip
   when Firefox is fixed). Sizing with it: a fixed delay a per chunk settles chunks at
   b·n = TARGET_CHUNK_MS − a (n_next = 30·n/(b·n + a)), so it only collapses to the floor when
   a ≥ 30 ms; a nudge every ~4 ms (also the browsers' clamp on nested timers) costs ~10%.
   Exact alternative: `timestamp-query` for real GPU time (Firefox 155 has it), optional with
   a fallback.
9. **The page swallows browser shortcuts and the context menu (web)**, reported 2026-10-07: in
   Firefox/Zen, right-click does nothing and Cmd-L (and likely other browser shortcuts) don't
   work over the canvas. Probably winit's web default `prevent_default = true` (canvas set up
   in `src/main.rs`, `with_canvas`), which cancels every keyboard, wheel and context-menu
   event. Possible fix: `with_prevent_default(false)` and cancel only the events the app uses.

## Maintenance

- **Move the pinned nightly forward** now and then ([update the toolchain](how-to/update-the-toolchain.md));
  `+atomics` is being phased out, so a newer one may need changes to the web build.
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
