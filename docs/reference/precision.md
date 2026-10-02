# Precision and floatexp

How far each number type reaches, and the floatexp pipeline that takes the GPU below f32's
exponent range. Why perturbation needs only low precision per pixel:
[perturbation](../explanation/perturbation.md).

## The number types

| where | type | why |
|---|---|---|
| view coordinates, reference orbits | `dashu` `FBig` (arbitrary precision) | exact positions at any depth; `pixel_to_coord` is exact |
| the reference orbit as uploaded | f32 (f32 pipeline) or floatexp (deep pipeline), both from an f64 copy (`Reference::orbit64`) | the f64 copy limits them to ~2⁻¹⁰²² |
| GPU deltas | f32, or floatexp below the threshold | speed |
| CPU deltas | `Pf` = **f32** (on purpose, to match the GPU in comparison tests; f64 would reach far deeper) | ask before changing |
| BLA table | `Fx`, a small floatexp on the CPU (`bla.rs`) | A overflows and R underflows f64 at depth |
| nucleus derivative | f64 mantissa + exponent (`Deriv`, `nucleus.rs`) | same |

## Why floatexp

f32 hits a hard wall at the **exponent**, not the mantissa: once units-per-pixel falls below
~2⁻¹²⁶ (≈ 1e−38) the seed δ₀ is subnormal and neighbouring pixels collapse to one value.
Double-single (hi + lo f32) would **not** help: it keeps the f32 exponent range.

**Floatexp** is a shared-exponent complex number, `Fe { m: vec2<f32>, e: i32 }`
(`shaders/floatexp.wgsl`).

## The deep pipeline

- **Chosen per pass:** floatexp when `upp_log2(depth) < FE_THRESHOLD` (−100), else f32.
- **Seeds:** `pack_fe(dx, dy, base_exp)` builds them in **pixel units** (O(1e4), so the f64
  math is always safe) and puts the depth scale 2^upp into the exponent. **Never form 2^upp
  as an f64 on the deep path**: it underflows.
- **Two phases:** δ in floatexp until its exponent `e > −60`, then plain f32 (δ₀ is negligible
  there). **δ can become tiny again** — at a zero of the reference orbit (a nucleus orbit hits
  0 once per period, where 2·X·δ vanishes and δ ← δ² + δ₀), after a rebase, or after a BLA
  jump: a phase-2 step whose result is below 2⁻⁶² is redone in floatexp and the pixel returns
  to phase 1. (Switching at 2⁻¹⁰⁰ without that fallback let δ underflow to exactly 0, and pixels
  followed the reference forever: black octagons and smeared streaks at 2⁻²²⁰. With the
  fallback, thresholds 2⁻⁶⁰…2⁻¹¹⁰ are equally accurate and 2⁻⁶⁰ is fastest.)
- **The reference orbit is uploaded as floatexp** for the deep pipeline (`Reference::orbit_fe`,
  `pack_fe` layout; `prepare(r, log2_dc, upp)` picks the format). A deep orbit passes far
  closer to 0 than f32 reaches (a nucleus comes within ~2⁻¹⁴⁹ / 2⁻²⁷¹ of 0 at its parents'
  periods, at 2⁻³¹⁴); flushed to 0 there, the 2·X·δ term vanished while δ was smaller still,
  and every pixel shadowed the reference: all black with a nucleus, or all "escaping" with a
  point reference (`view_2026_09_23c_matches_exact`).

## The remaining limit: the mantissa

f32 rounding makes a fraction of long-lived near-boundary pixels escape a few iterations off
(`diag_reference_precision`, ignored test): at the 2⁻³⁰⁵ view, 611 of 15 000 samples off, 10 by
more than 50 iterations, with BLA (5946 / 112 without). Mostly from rounding the reference
orbit to f32, but a hi + lo reference alone does not fix it. Double-single delta arithmetic or
accepting f32 is an open decision ([backlog](../backlog.md)).
