# Perturbation, rebasing, BLA, interior detection

The math the renderer rests on. Where it is implemented: [GPU backend](../reference/gpu-backend.md),
[precision](../reference/precision.md).

## Perturbation

Deep views need hundreds of digits to tell neighbouring pixels apart, and iterating every pixel
in arbitrary precision is hopeless. Perturbation iterates **one** point exactly and every pixel
as a small difference from it.

Take a reference `C` with orbit `X₀ = C, X_{n+1} = Xₙ² + C`. A pixel at `c = C + δ₀` has
`zₙ = Xₙ + δₙ` with

```
δ_{n+1} = 2·Xₙ·δₙ + δₙ² + δ₀
```

and escapes when `|Xₙ + δₙ|² > 4`. (This code's convention is `z₀ = c`, not 0.) Only the
reference needs high precision (`FBig`); δ fits in f32 (or [floatexp](../reference/precision.md)
at depth).

**Perturbation is algebraically exact for any size of δ.** A large δ only risks float
precision, so there is **no** |δ|/|X| (Pauldelbrot-style) glitch heuristic: a pixel is
"glitched" only when the reference orbit ended early (`!is_full`, the reference escaped) and the
pixel had not escaped yet. This matches `check_divergence_delta` (the CPU's reference delta
loop). An earlier `|δ|² > 1e−6·|x|²` check in the shader flagged nearly every pixel at low zoom
and drew an "iteration-1 egg".

## Why a nucleus as the reference

A reference orbit that escapes leaves every pixel that lives longer "glitched". A **nucleus**
(the centre of a minibrot or bulb, period p: z returns to 0 every p iterations) never escapes,
so with it normally nothing glitches. Finding one is a Newton iteration from near the view; the
subtleties (precision, caching, searching in the background) are in the
[GPU backend](../reference/gpu-backend.md#references).

## Rebasing

Zhuoran (2021): when `|z| < |δ|`, set `δ ← z² + δ₀` and restart the reference index at 0 — the
`X₀ = C` form of "δ ← z, back to the start". It is exact, not a heuristic, and removes the
blocky precision-loss artifacts that appear once the reference is a nucleus the pixel's orbit
does not follow. The shaders track `n` (iterations, reported) and `m` (reference index)
separately.

## BLA (bilinear approximation)

While |δ| is tiny next to |2X|, a step is linear: `δ ← A·δ + B·δ₀`. Blocks of 2^k steps compose
into one (A, B) with a validity radius R (relative error ≤ 2⁻²⁴):

```
A = A_y·A_x,   B = A_y·B_x + B_y,   R = min(R_x, (R_y − |B_x|·max|δ₀|) / |A_x|)
single step: R = ε·|2X|
```

- The table is a binary tree over the reference orbit (level k: blocks at multiples of 2^k,
  stored at index `2^(L+1) − 2^(L+1−k)`), built on the CPU in a small floatexp (`Fx`) per pass
  and reference, with R bounded by the pass's max |δ₀| (`bla.rs`).
- The shaders search upwards from level 1 (a block's R ≤ its first half's, so the first invalid
  level ends the search; level 0 is never used), with exponential back-off after failed
  searches (reset on a jump or rebase); they apply the block in floatexp, add log|A|² to the
  interior-detection derivative, and never jump across an interior window boundary or the
  iteration limit.
- **No escape is skipped**: within R, |z| ≤ |X|·(1 + 2⁻²³), and single steps with |X| near or
  above 2 (an escaping reference's end) are invalid, so no block spans them. log2|δ| is clamped
  above the invalid radius (at δ = 0 it is −∞, which passed invalid blocks: a pixel at its own
  reference jumped over its escape).
- Measured: the 2⁻³⁰⁵ view went from ~2 min to ~1.5 s per generation and became more accurate
  (fewer f32 rounding steps); shallow views ~5–10 % slower. `USE_BLA` switches it off for A/B.

## Interior detection

Pixels in the set run to the iteration limit — the most expensive pixels there are. With a
nucleus reference, the shaders track log2|dz/dz₀|² and, every window of ≥ 128 iterations (a
whole number of the nucleus period p), compare it with the previous window: **2 consecutive
windows contracting by ≥ 0.9× per period → in the set**. Inside a component the cycle
multiplier |λ| < 1; just outside |λ| ≥ 1.

- Short windows (< ~64) produce false positives (escaping pixels contracting briefly near 0):
  don't shorten them.
- **A window also needs z back** within 2⁻¹⁰ (relative) of where it was at the previous
  boundary (`INTERIOR_RETURN`). With a nucleus of another period than a minibrot in view (a
  cached one from further out), windows are out of phase with the pixel's cycle and escaping
  pixels looked contracting: black discs, misshapen minibrots (`view_2026_09_25_black_disc`:
  519 black vs 378 exact, now 359 as without detection; `diag_view_2026_09_25b_references`,
  p = 55: 129 falsely in the set, now 8 as without). Pixels near a minibrot of another period
  than the reference now run to the iteration limit instead: correct, slower.
- The check runs at the top of the loop at each boundary (z_n = X_m + δ, the same phase after a
  step or a BLA jump).
- Cost: shallow cardioid work 16.9 → 19.4 % of full iterations; other
  `diag_interior_detection` views unchanged; no false positives. Parameters: the `INTERIOR_*`
  consts in `gpu.rs`, passed as uniforms.
