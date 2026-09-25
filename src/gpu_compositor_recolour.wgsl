// Tile colouring: tiles' iteration counts → one mip level of their colour
// textures. Runs whenever a tile's contents or the palette change, so a
// palette change recolours on the GPU without touching the CPU tile data.
// Tiles of one shape share 2D array textures (a chunk, one layer per tile);
// a dispatch colours the layers listed in `jobs`, one per z.
//
// `iters` holds the tile's data texels (the pass-stride grid, or the whole
// tile from the first sub-pass on), prepared by `iteration_texels`: an
// escape iteration (0 = in the set, drawn black), or GAP for a sub-pass pixel
// not computed yet, filled with the mean of its computed axis neighbours.
// Each output texel is the average, in linear light, of the 2^k × 2^k data
// texels it covers (k = the level); the texture is written as sRGB-encoded
// bytes (storage textures can't be sRGB) and sampled through an sRGB view.

struct Palette {
    hue:   f32,   // phase offsets in turns (`PalettePhase`)
    light: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> pal: Palette;
@group(0) @binding(1) var<storage, read> jobs: array<u32>;      // layers (dynamic offset per dispatch)
@group(1) @binding(0) var iters: texture_2d_array<u32>;
@group(1) @binding(1) var out: texture_storage_2d_array<rgba8unorm, write>;

const GAP: u32 = 0x80000000u;

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3<f32>(0.0031308));
}

const TAU: f32 = 6.283185307179586;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

/// The `hsl` crate's hue_to_rgb.
fn hue_to_rgb(p: f32, q: f32, t0: f32) -> f32 {
    var t = t0;
    if (t < 0.0) { t += 1.0; } else if (t > 1.0) { t -= 1.0; }
    if (t < 1.0 / 6.0) { return p + (q - p) * 6.0 * t; }
    if (t < 0.5) { return q; }
    if (t < 2.0 / 3.0) { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}

/// Linear colour of an escape iteration (0 = in the set: black). This is
/// `val_to_color` (rendering.rs) with the `hsl` crate's HSL → RGB, rounded
/// to bytes like it; the two must be kept in sync (`recolour_matches_cpu`).
/// Computed rather than looked up in a table: the lookup is a gather by the
/// pixels' iterations, which on noisy tiles was the pass's main cost
/// (`diag_recolour_speed`, s = 4 screen: 155 ms from a storage buffer,
/// 214 ms from a texture, 25 ms computed; smooth tiles ~20 ms either way).
/// f32 argument precision: exact to ~2^-24 relative, so at 10^7 iterations
/// the lightness wave's phase is off by ~0.002 turns (invisible).
fn colour(v: u32) -> vec3<f32> {
    if (v == 0u) { return vec3<f32>(0.0); }
    let n = f32(v);
    let h = sin(n / 1200.0 + TAU * pal.hue) * 0.5;   // turns: sin·180° / 360°
    let s = 0.7;
    let l = sin(n / 40.0 + TAU * pal.light) * 0.3 + 0.4;
    let q = select(l + s - l * s, l * (1.0 + s), l < 0.5);
    let p = 2.0 * l - q;
    let rgb = vec3<f32>(hue_to_rgb(p, q, h + 1.0 / 3.0), hue_to_rgb(p, q, h), hue_to_rgb(p, q, h - 1.0 / 3.0));
    return srgb_to_linear(round(rgb * 255.0) / 255.0);
}

/// Add the colour at `p` to `sum` if it is inside the tile and computed.
fn add_known(p: vec2<i32>, layer: u32, n: i32, sum: ptr<function, vec3<f32>>, k: ptr<function, f32>) {
    if (any(p < vec2<i32>(0)) || any(p >= vec2<i32>(n))) { return; }
    let v = textureLoad(iters, p, layer, 0).r;
    if ((v & GAP) != 0u) { return; }
    *sum += colour(v);
    *k += 1.0;
}

/// Linear colour of data texel `p`; a gap is its computed axis neighbours'
/// mean (the sub-pass order guarantees all four, fewer at the tile edge).
fn texel(p: vec2<i32>, layer: u32, n: i32) -> vec3<f32> {
    let v = textureLoad(iters, p, layer, 0).r;
    if ((v & GAP) == 0u) { return colour(v); }
    var sum = vec3<f32>(0.0);
    var k = 0.0;
    add_known(p + vec2<i32>(-1, 0), layer, n, &sum, &k);
    add_known(p + vec2<i32>(1, 0), layer, n, &sum, &k);
    add_known(p + vec2<i32>(0, -1), layer, n, &sum, &k);
    add_known(p + vec2<i32>(0, 1), layer, n, &sum, &k);
    return sum / max(k, 1.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let layer = jobs[id.z];
    let n = i32(textureDimensions(iters).x);
    let f = n / i32(size.x);   // data texels per output texel, per axis
    let origin = vec2<i32>(id.xy) * f;
    var sum = vec3<f32>(0.0);
    for (var y = 0; y < f; y++) {
        for (var x = 0; x < f; x++) {
            sum += texel(origin + vec2<i32>(x, y), layer, n);
        }
    }
    let lin = sum / f32(f * f);
    textureStore(out, vec2<i32>(id.xy), layer, vec4<f32>(linear_to_srgb(lin), 1.0));
}
