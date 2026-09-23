// Deep-zoom perturbation shader.  Identical in result to perturbation.wgsl,
// but the per-pixel delta is iterated in shared-exponent floatexp (see
// floatexp.wgsl, prepended at compile time) so it does not underflow when the
// zoom passes the f32 exponent floor.
//
// Two phases per pixel: iterate in floatexp while the delta is tiny; once it
// is comfortably inside the f32 range (> 2^F32_SWITCH_EXP), continue in plain
// f32, where delta_0 is negligible (and underflows to 0).
//
// δ can become tiny again, though: at a zero of the reference orbit (a
// nucleus orbit hits 0 once per period) the 2·X·δ term vanishes and
// δ ← δ² + δ₀, and a rebase sets δ ← z² + δ₀. In f32 both terms would
// underflow to exactly 0 and the pixel would silently follow the reference
// forever (a nucleus: never escapes). So a phase-2 step whose result would
// drop below F32_MIN_DELTA is redone in floatexp and the pixel goes back to
// phase 1. (Seen as black octagons and smeared streaks at 2^-220 while
// switching at 2^-100.)
//
// Both phases rebase as in perturbation.wgsl: when |z| < |δ|, δ ← z² + δ_0 and
// the reference index restarts at 0.  `n` counts iterations, `m` indexes the
// reference orbit.  Interior detection is as in perturbation.wgsl, with
// log2|z|² taken from the floatexp z in phase 1 (z can be far below f32 there).
// BLA jumps (perturb_common.wgsl) work in both phases; a jump that leaves δ
// tiny in phase 2 goes back to phase 1, like any other tiny result.
//
// The reference orbit is floatexp too: a deep orbit passes far closer to 0
// than f32 reaches (~2^-149 and ~2^-271 at 2^-314), and while δ is smaller
// still, 2·X·δ is what separates the pixel from the reference. Flushed to 0,
// every pixel shadowed the reference and none escaped (all black at 2^-314).

struct Uniforms {
    pixel_count : u32,
    orbit_len   : u32,
    is_full     : u32,
    dispatch_w  : u32,
    // Interior detection (see INTERIOR_* in gpu.rs); window 0 disables it.
    interior_window      : u32,
    interior_contraction : f32,
    interior_windows     : u32,
    // BLA table (see perturb_common.wgsl); bla_levels 0 disables it.
    bla_levels           : u32,
    bla_log2_size        : u32,
    _pad0                : u32,
    _pad1                : u32,
    _pad2                : u32,
}

@group(0) @binding(0) var<uniform>             uniforms     : Uniforms;
// floatexp seed: xy = complex mantissa, z = exponent (as f32), w unused.
@group(0) @binding(1) var<storage, read>       pixel_deltas : array<vec4<f32>>;
// floatexp reference orbit, same packing as the seeds.
@group(0) @binding(2) var<storage, read>       orbit_data   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output       : array<u32>;

const GLITCH_BIT : u32 = 0x80000000u;
// Once the delta's exponent exceeds this it is safely a normal f32, so we can
// finish in plain f32. 2^-100 is far above the f32 floor (2^-126).
const F32_SWITCH_EXP : i32 = -60;
// 2^-62: below this a phase-2 step goes back to floatexp (δ² stays a normal
// f32 above it: 2^-124 > 2^-126).
const F32_MIN_DELTA : f32 = 2.168404344971009e-19;

// Reference orbit value X_m (unnormalized mantissa: fine for fe_to_c).
fn orbit_raw(m : u32) -> Fe {
    let o = orbit_data[m];
    return Fe(o.xy, i32(o.z));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.y * uniforms.dispatch_w + gid.x;
    if (idx >= uniforms.pixel_count) { return; }

    let seed = pixel_deltas[idx];
    let d0    = fe_norm(Fe(seed.xy, i32(seed.z)));
    var delta = d0;

    var n = 0u;  // iteration count
    var m = 0u;  // index into the reference orbit (resets on rebase)
    var backoff = BlaBackoff(0u, 1u);
    var ld = 0.0;  // log2 |dz/dz_0|²
    var ld_prev = 0.0;
    var have_prev = false;
    var streak = 0u;

    loop {
    // Phase 1: floatexp while delta is too small for f32.
    loop {
        if (n >= uniforms.orbit_len) { break; }
        var hit = BlaHit(0u, 0u);
        if (bla_should_search(&backoff, m)) {
            hit = bla_find(m, 0.5 * log2(dot(delta.m, delta.m)) + f32(delta.e), jump_limit(n));
            if (hit.len == 0u) { bla_failed(&backoff); } else { bla_reset(&backoff); }
        }
        if (hit.len != 0u) {
            let e = bla_table[hit.idx];
            delta = bla_apply(e, delta, d0);
            if (uniforms.interior_window != 0u) {
                ld += bla_log2_a2(e);
                if (interior_check(n + hit.len - 1u, &ld_prev, &have_prev, &streak, ld)) {
                    output[idx] = 0u;
                    return;
                }
            }
            n = n + hit.len;
            m = m + hit.len;
            if (delta.e > F32_SWITCH_EXP) { break; }
            continue;
        }
        let cf = fe_norm(orbit_raw(m));
        let x = fe_to_c(cf) + fe_to_c(delta);
        if (dot(x, x) > 4.0) {
            output[idx] = n + 1u;
            return;
        }
        // z in floatexp: X can be ~0 (a nucleus orbit hits 0), where the f32
        // value above has underflowed.
        let z = fe_add(cf, delta);
        if (uniforms.interior_window != 0u) {
            ld += 2.0 + log2(dot(z.m, z.m)) + 2.0 * f32(z.e);
            if (interior_check(n, &ld_prev, &have_prev, &streak, ld)) {
                output[idx] = 0u;
                return;
            }
        }
        if (fe_abs_lt(z, delta)) {
            // Rebase: δ ← z² + δ_0, back to the start of the reference.
            delta = fe_add(fe_mul(z, z), d0);
            m = 0u;
            bla_reset(&backoff);
        } else {
            // δ_{n+1} = 2 X_n δ_n + δ_n² + δ_0
            let t1 = fe_mul(delta, Fe(cf.m, cf.e + 1));
            let t2 = fe_mul(delta, delta);
            delta = fe_add(fe_add(t1, t2), d0);
            m = m + 1u;
        }
        n = n + 1u;
        if (delta.e > F32_SWITCH_EXP) { break; }
    }

    if (n >= uniforms.orbit_len) { break; }

    // Phase 2: plain f32 (delta now normal; d0 underflows to ~0).
    var df  = fe_to_c(delta);
    let d0f = fe_to_c(d0);
    loop {
        if (n >= uniforms.orbit_len) { break; }
        var hit = BlaHit(0u, 0u);
        if (bla_should_search(&backoff, m)) {
            hit = bla_find(m, 0.5 * log2(dot(df, df)), jump_limit(n));
            if (hit.len == 0u) { bla_failed(&backoff); } else { bla_reset(&backoff); }
        }
        if (hit.len != 0u) {
            let e = bla_table[hit.idx];
            let dn = bla_apply(e, fe_from_c(df), d0);
            if (uniforms.interior_window != 0u) {
                ld += bla_log2_a2(e);
                if (interior_check(n + hit.len - 1u, &ld_prev, &have_prev, &streak, ld)) {
                    output[idx] = 0u;
                    return;
                }
            }
            n = n + hit.len;
            m = m + hit.len;
            if (dn.e <= F32_SWITCH_EXP) {
                delta = dn;   // tiny again: back to phase 1
                break;
            }
            df = fe_to_c(dn);
            continue;
        }
        let c = fe_to_c(orbit_raw(m));
        let x = c + df;
        let x2 = dot(x, x);
        if (x2 > 4.0) {
            output[idx] = n + 1u;
            return;
        }
        if (uniforms.interior_window != 0u) {
            ld += 2.0 + log2(x2);
            if (interior_check(n, &ld_prev, &have_prev, &streak, ld)) {
                output[idx] = 0u;
                return;
            }
        }
        let rebase = x2 < dot(df, df);
        var next : vec2<f32>;
        if (rebase) {
            // Rebase: δ ← z² + δ_0, back to the start of the reference.
            next = vec2<f32>(x.x * x.x - x.y * x.y, 2.0 * x.x * x.y) + d0f;
        } else {
            let re = 2.0 * c.x * df.x - 2.0 * c.y * df.y
                   + df.x * df.x      - df.y * df.y
                   + d0f.x;
            let im = 2.0 * c.x * df.y + 2.0 * c.y * df.x
                   + 2.0 * df.x * df.y
                   + d0f.y;
            next = vec2<f32>(re, im);
        }
        let m_prev = m;
        m = select(m + 1u, 0u, rebase);
        n = n + 1u;
        if (rebase) { bla_reset(&backoff); }
        if (max(abs(next.x), abs(next.y)) < F32_MIN_DELTA) {
            // δ is tiny again: redo this step in floatexp (z formed there
            // too, as X + δ may cancel below f32 range) and go back to phase 1.
            let dfe = fe_from_c(df);
            let cf = fe_norm(orbit_raw(m_prev));
            if (rebase) {
                let z = fe_add(cf, dfe);
                delta = fe_add(fe_mul(z, z), d0);
            } else {
                delta = fe_add(fe_add(fe_mul(dfe, Fe(cf.m, cf.e + 1)), fe_mul(dfe, dfe)), d0);
            }
            break;
        }
        df = next;
    }
    if (n >= uniforms.orbit_len) { break; }
    }

    if (uniforms.is_full != 0u) {
        output[idx] = 0u;
    } else {
        output[idx] = GLITCH_BIT | uniforms.orbit_len;
    }
}
