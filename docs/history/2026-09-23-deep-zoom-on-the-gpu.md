# 2026-09-23 – 24: deep zoom on the GPU

Two days that took the GPU backend from artifacts past 2⁻²²⁰ to accurate views at 2⁻³²⁰ in
seconds. Every fix started from a view the owner reported (a Cmd-C string), reproduced in a test
first. The current facts are in the reference ([GPU backend](../reference/gpu-backend.md),
[precision](../reference/precision.md), [perturbation](../explanation/perturbation.md)); this is
the order and the reasons.

## 2026-09-23

- **Dependencies upgraded**, GPU diagnostics (`DIAG`), an interruptible parallel exact resolve
  of leftover glitches.
- **A nearby nucleus as the reference** (`nucleus.rs`): its orbit never escapes, so normally
  nothing glitches.
- **Time-bounded chunked dispatch**, interruptible between chunks.
- **The final pass split into three sub-passes** (cell centres first); **quadtree seeding**
  across depths.
- `CLAUDE.md` project notes started (2026-09-23; archived verbatim in
  [working notes](working-notes-2026-10-03.md)).
- **The progress bar** became a proportional white bar with momentum.
- **Rebasing** in both shaders. The **title diagnostics** for keys. **Cmd/Ctrl-C/V on the
  layout's character** (the owner types Dvorak).
- **Tiles store escape iterations** instead of colours, so changing iterations keeps work.
- **Interior detection** via the cycle multiplier.
- **Floatexp fix:** pixels lost their delta at the reference orbit's zeros (black octagons and
  streaks at 2⁻²²⁰): a phase-2 step that underflows goes back to floatexp.
- **macOS kills long command buffers**, silently: dispatches kept short, the `NOT_RUN` sentinel
  and retries added.
- **BLA**: the 2⁻³⁰⁵ view went from ~2 min to ~1.5 s per generation.
- **An open bug — a uniform image past 2⁻³¹⁰** — was recorded, reproduced
  (`view_2026_09_23c_matches_exact`) and fixed the same day: the deep pipeline now gets the
  reference orbit in **floatexp** (flushed to f32 it hit 0 and every pixel shadowed the
  reference).

## 2026-09-24

- Pixels at or very near an escaping reference escape with it (BLA had jumped over the escape).
- **Nucleus search precision** for deep minibrots (raise to 2·log2|dz/dc| + 64 bits; the
  "nucleus" orbit had escaped after ~1.1 periods), and faster (doubling same-direction steps).
- **Failed searches remembered** for nearby views.
- **A cached nucleus reused up to 1024 view radii away** (`diag_far_reference`); the limit is
  f32 pixel-offset quantization, not the reference.
- **The open threads written down**; the owner **rejected ring-by-ring refinement** — render
  order stays pass-major.
- **CPU/GPU overlap**: the next chunk packed and submitted while the current one is
  post-processed (~17 % faster per generation). Two chunks queued at once was tried: Metal ran
  them concurrently, slower, unmeasurable — rejected.
- **Background nucleus search** (agreed with the owner that morning): render at once on a
  provisional reference, defer the glitched pixels. First pixels after ~110 ms instead of
  1–2 s.
- **Start on the GPU backend.**
- **The sampling ratio s** (←/→) and **antialiasing** by exact area averaging in linear light.
  Considered first for "compute fewer pixels": choosing the depth by rounding instead of
  ceiling (~2.3× fewer pixels) — superseded by s, which the owner can set.
- **Mip levels only as needed**, uploaded only for the current s.
- **The tile budget follows the view** (3× its tiles, 1 GiB floor): with a fixed 1 GiB, s = 4's
  view alone filled it.
- **Sub-pass gaps** filled with the neighbours' mean in linear light.
