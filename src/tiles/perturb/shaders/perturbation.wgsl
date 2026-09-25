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
    interior_return      : f32,
    _pad1                : u32,
    _pad2                : u32,
}

@group(0) @binding(0) var<uniform>             uniforms     : Uniforms;
@group(0) @binding(1) var<storage, read>       pixel_deltas : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       orbit_data   : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output       : array<u32>;

// Bit 31 set means glitched; low 31 bits carry the iteration when detected.
const GLITCH_BIT : u32 = 0x80000000u;

// Perturbation is algebraically exact: X_n = ref_n + delta_n is the true orbit
// value for any size of delta, so there is no |delta|/|X| glitch heuristic.  A
// pixel is only "glitched" when the reference orbit ended early (is_full == 0)
// and the pixel had not yet escaped; such pixels get a fresh reference.
//
// Rebasing (Zhuoran 2021): when the pixel's value z = X + δ becomes smaller
// than δ, the pixel's orbit has come close to 0 while the reference's has not,
// and continuing would lose precision as δ ≈ -X cancels.  Instead, restart
// against the start of the reference orbit: in the X_0 = C convention the next
// value is z² + c = X_0 + (z² + δ_0), so δ ← z² + δ_0 at reference index 0.
// This is exact too; it only changes which reference point δ is measured from.
//
// Interior detection: track ld = log2|dz/dz_0|² (|dz_{n+1}|² = 4|z_n|²|dz_n|²,
// additive in log space so it cannot over/underflow). Every `interior_window`
// iterations (a whole number of the nucleus period) compare it with the
// previous window (and require z to have come back, see interior_check):
// inside a component the cycle multiplier |λ| < 1, so it
// shrinks; `interior_windows` consecutive shrinks by `interior_contraction`
// declare the pixel in the set, instead of running to the iteration limit.
//
// BLA: while δ is tiny next to X, skip a whole block of iterations at once
// (δ ← A·δ + B·δ₀, in floatexp since A can be huge); see perturb_common.wgsl.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.y * uniforms.dispatch_w + gid.x;
    if idx >= uniforms.pixel_count { return; }

    let d0    = pixel_deltas[idx];
    var delta = d0;
    var m     = 0u;  // index into the reference orbit (resets on rebase)
    var ld     = 0.0; // log2 |dz/dz_0|²
    var interior = interior_new();
    var n = 0u;
    var backoff = BlaBackoff(0u, 1u);

    loop {
        if n >= uniforms.orbit_len { break; }
        // Every n is reached here once (BLA jumps stop at window boundaries).
        if interior_boundary(n) && interior_check(orbit_data[m] + delta, ld, &interior) {
            output[idx] = 0u;
            return;
        }

        var hit = BlaHit(0u, 0u);
        if bla_should_search(&backoff, m) {
            hit = bla_find(m, 0.5 * log2(dot(delta, delta)), jump_limit(n));
            if hit.len == 0u { bla_failed(&backoff); } else { bla_reset(&backoff); }
        }
        if hit.len != 0u {
            let e = bla_table[hit.idx];
            delta = fe_to_c(bla_apply(e, fe_from_c(delta), fe_from_c(d0)));
            if uniforms.interior_window != 0u { ld += bla_log2_a2(e); }
            n += hit.len;
            m += hit.len;
            continue;
        }

        let c = orbit_data[m];
        let x = c + delta;
        let x2 = dot(x, x);

        if x2 > 4.0 {
            output[idx] = n + 1u;
            return;
        }

        if uniforms.interior_window != 0u { ld += 2.0 + log2(x2); }

        if x2 < dot(delta, delta) {
            // Rebase: δ ← z² + δ_0, back to the start of the reference.
            delta = vec2<f32>(x.x * x.x - x.y * x.y, 2.0 * x.x * x.y) + d0;
            m = 0u;
            bla_reset(&backoff);
        } else {
            // δ_{n+1} = 2 X_n δ_n + δ_n² + δ_0
            let re = 2.0 * c.x * delta.x - 2.0 * c.y * delta.y
                   + delta.x * delta.x    - delta.y * delta.y
                   + d0.x;
            let im = 2.0 * c.x * delta.y + 2.0 * c.y * delta.x
                   + 2.0 * delta.x * delta.y
                   + d0.y;
            delta = vec2<f32>(re, im);
            m = m + 1u;
        }
        n += 1u;
    }

    if uniforms.is_full != 0u {
        output[idx] = 0u;
    } else {
        output[idx] = GLITCH_BIT | uniforms.orbit_len;
    }
}
