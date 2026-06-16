// Deep-zoom perturbation shader.  Identical in result to perturbation.wgsl,
// but the per-pixel delta is iterated in shared-exponent floatexp (see
// floatexp.wgsl, prepended at compile time) so it does not underflow when the
// zoom passes the f32 exponent floor.
//
// Two phases per pixel: iterate in floatexp while the delta is tiny; once it
// grows into the normal f32 range, drop to plain f32 for the rest (the seed
// delta_0 is negligible by then, exactly as the f64 CPU path treats it).

struct Uniforms {
    pixel_count : u32,
    orbit_len   : u32,
    is_full     : u32,
    dispatch_w  : u32,
}

@group(0) @binding(0) var<uniform>             uniforms     : Uniforms;
// floatexp seed: xy = complex mantissa, z = exponent (as f32), w unused.
@group(0) @binding(1) var<storage, read>       pixel_deltas : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       orbit_data   : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output       : array<u32>;

const GLITCH_BIT : u32 = 0x80000000u;
// Once the delta's exponent exceeds this it is safely a normal f32, so we can
// finish in plain f32. 2^-100 is far above the f32 floor (2^-126).
const F32_SWITCH_EXP : i32 = -100;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.y * uniforms.dispatch_w + gid.x;
    if (idx >= uniforms.pixel_count) { return; }

    let seed = pixel_deltas[idx];
    let d0    = fe_norm(Fe(seed.xy, i32(seed.z)));
    var delta = d0;

    var i = 0u;

    // Phase 1: floatexp while delta is too small for f32.
    loop {
        if (i >= uniforms.orbit_len) { break; }
        let c = orbit_data[i];
        let x = c + fe_to_c(delta);
        if (dot(x, x) > 4.0) {
            output[idx] = i + 1u;
            return;
        }
        // δ_{n+1} = 2 X_n δ_n + δ_n² + δ_0
        let t1 = fe_mul(delta, fe_from_c(2.0 * c));
        let t2 = fe_mul(delta, delta);
        delta = fe_add(fe_add(t1, t2), d0);
        i = i + 1u;
        if (delta.e > F32_SWITCH_EXP) { break; }
    }

    // Phase 2: plain f32 (delta now normal; d0 underflows to ~0, as on the CPU).
    var df  = fe_to_c(delta);
    let d0f = fe_to_c(d0);
    loop {
        if (i >= uniforms.orbit_len) { break; }
        let c = orbit_data[i];
        let x = c + df;
        if (dot(x, x) > 4.0) {
            output[idx] = i + 1u;
            return;
        }
        let re = 2.0 * c.x * df.x - 2.0 * c.y * df.y
               + df.x * df.x      - df.y * df.y
               + d0f.x;
        let im = 2.0 * c.x * df.y + 2.0 * c.y * df.x
               + 2.0 * df.x * df.y
               + d0f.y;
        df = vec2<f32>(re, im);
        i = i + 1u;
    }

    if (uniforms.is_full != 0u) {
        output[idx] = 0u;
    } else {
        output[idx] = GLITCH_BIT | uniforms.orbit_len;
    }
}
