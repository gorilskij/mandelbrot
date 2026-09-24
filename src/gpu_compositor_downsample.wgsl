// Final compositor pass: the offscreen image (tiles drawn 1:1 at the mip level
// that leaves a ratio q in [1, 2) tile texels per screen pixel, see
// gpu_compositor.rs) is averaged down to the screen.
//
// A screen pixel covers a q×q square of offscreen texels, starting at
// floor(pixel)·q + frac. For q >= 1 its colour is the exact area average of
// that square (box filter: each texel weighted by how much of it is covered,
// at most 3×3 texels since q < 2). For q < 1 (sampling ratio below 1) the
// image is magnified instead, bilinearly. The offscreen image is sRGB, so
// loads and filtering are in linear light; the result is encoded back to sRGB
// by hand because the surface format is not an sRGB one.

// 16 bytes, as written by gpu_compositor.rs: [q, 0, frac.x, frac.y]
// (vec2 is 8-byte aligned, so the padding comes before it).
struct Params {
    q    : f32,
    _pad : f32,
    frac : vec2<f32>,
}

@group(0) @binding(0) var img : texture_2d<f32>;
@group(0) @binding(1) var smp : sampler;
@group(0) @binding(2) var<uniform> p : Params;

@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
    // One triangle covering the screen.
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn linear_to_srgb(c : vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0031308)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// Coverage of texel [t, t+1) by [a, a+q).
fn overlap(t : f32, a : f32, q : f32) -> f32 {
    return max(0.0, min(t + 1.0, a + q) - max(t, a));
}

@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(img));
    let a = floor(pos.xy) * p.q + p.frac;
    var c = vec3<f32>(0.0);
    if (p.q < 1.0) {
        c = textureSampleLevel(img, smp, (a + 0.5 * p.q) / vec2<f32>(dims), 0.0).rgb;
    } else {
        let t0 = floor(a);
        for (var dy = 0; dy < 3; dy++) {
            let ty = t0.y + f32(dy);
            let wy = overlap(ty, a.y, p.q);
            if (wy <= 0.0) { continue; }
            for (var dx = 0; dx < 3; dx++) {
                let tx = t0.x + f32(dx);
                let wx = overlap(tx, a.x, p.q);
                if (wx <= 0.0) { continue; }
                let at = clamp(vec2<i32>(i32(tx), i32(ty)), vec2<i32>(0), dims - 1);
                c += textureLoad(img, at, 0).rgb * (wx * wy);
            }
        }
        c /= p.q * p.q;
    }
    return vec4<f32>(linear_to_srgb(c), 1.0);
}
