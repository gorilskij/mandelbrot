// Shared by both perturbation shaders (prepended after floatexp.wgsl, before
// the shader body, which declares `uniforms`).

// ---------------------------------------------------------------------------
// BLA: bilinear approximation (see bla.rs for the table and the maths)
// ---------------------------------------------------------------------------

// One block of 2^k steps: δ_after = A·δ + B·δ₀, valid while |δ| < 2^log2_r.
// A and B are floatexp (normalized mantissa, exponent). 32 bytes, same
// layout as BlaEntry in bla.rs (vec2s first: they are 8-byte aligned).
struct Bla {
    a      : vec2<f32>,
    b      : vec2<f32>,
    a_e    : i32,
    b_e    : i32,
    log2_r : f32,
    _pad   : u32,
}

@group(0) @binding(4) var<storage, read> bla_table : array<Bla>;

struct BlaHit {
    idx : u32,
    len : u32,   // 0 = no valid block
}

// The longest valid block starting at reference index m for |δ| = 2^log2_d,
// at most max_len steps. Level k holds the blocks starting at multiples of
// 2^k, at index 2^(L+1) − 2^(L+1−k) + (m >> k). Level 0 (one step) is not
// used: a plain iteration costs the same and also checks escape/rebase.
//
// Searched upwards: a block's radius is at most its first half's (R =
// min(R_x, …)), so once a level is invalid no longer block at m can be
// valid. The common case, no valid block, costs one lookup (none for odd m).
//
// log2_d is clamped above the invalid radius (-1e30, finite because of fast
// math): at δ = 0 it is log2(0) = -inf, which passed invalid blocks too.
fn bla_find(m : u32, log2_d_raw : f32, max_len : u32) -> BlaHit {
    let log2_d = max(log2_d_raw, -1e29);
    let size_log2 = uniforms.bla_log2_size;
    var best = BlaHit(0u, 0u);
    if (uniforms.bla_levels < 2u || m >= (1u << size_log2)) { return best; }
    var k = 1u;
    loop {
        let len = 1u << k;
        if (k >= uniforms.bla_levels || (m & (len - 1u)) != 0u || len > max_len) { break; }
        let idx = (2u << size_log2) - (2u << (size_log2 - k)) + (m >> k);
        if (!(log2_d < bla_table[idx].log2_r)) { break; }
        best = BlaHit(idx, len);
        k = k + 1u;
    }
    return best;
}

// Back-off between BLA searches: after a failed search wait `wait` more
// iterations (1, 2, 4, … up to BLA_MAX_BACKOFF) before searching again;
// reset by a successful jump or a rebase (δ is small again). δ mostly grows
// between rebases, so repeated searches mostly fail; skipping one only costs
// speed, never accuracy.
const BLA_MAX_BACKOFF : u32 = 64u;

struct BlaBackoff {
    wait : u32,
    next : u32,
}

// Whether to search at this iteration (odd m never has a block).
fn bla_should_search(b : ptr<function, BlaBackoff>, m : u32) -> bool {
    if ((*b).wait > 0u) { (*b).wait = (*b).wait - 1u; return false; }
    return (m & 1u) == 0u;
}

fn bla_failed(b : ptr<function, BlaBackoff>) {
    (*b).wait = (*b).next;
    (*b).next = min((*b).next * 2u, BLA_MAX_BACKOFF);
}

fn bla_reset(b : ptr<function, BlaBackoff>) {
    (*b).wait = 0u;
    (*b).next = 1u;
}

// Apply block `e` to δ (both floatexp).
fn bla_apply(e : Bla, d : Fe, d0 : Fe) -> Fe {
    return fe_add(fe_mul(Fe(e.a, e.a_e), d), fe_mul(Fe(e.b, e.b_e), d0));
}

// log2 |A|² of block `e`: the block's factor on |dz/dz₀|², since
// A = Π 2·X_k ≈ Π 2·z_k over the block.
fn bla_log2_a2(e : Bla) -> f32 {
    return log2(dot(e.a, e.a)) + 2.0 * f32(e.a_e);
}

// Longest jump allowed from iteration count n: not past the iteration limit
// and not across an interior-detection window boundary (checks happen at
// window ends).
fn jump_limit(n : u32) -> u32 {
    var lim = uniforms.orbit_len - n;
    if (uniforms.interior_window != 0u) {
        lim = min(lim, uniforms.interior_window - n % uniforms.interior_window);
    }
    return lim;
}

// ---------------------------------------------------------------------------
// Interior detection (see INTERIOR_* in gpu.rs)
// ---------------------------------------------------------------------------

// Called after iteration n with ld = log2|dz/dz₀|². At the end of each
// interior window, update the contraction streak; true once enough
// consecutive windows contracted (pixel is in the set).
fn interior_check(n : u32, ld_prev : ptr<function, f32>, have_prev : ptr<function, bool>,
                  streak : ptr<function, u32>, ld : f32) -> bool {
    if ((n + 1u) % uniforms.interior_window != 0u) { return false; }
    if (*have_prev && ld - *ld_prev < uniforms.interior_contraction) {
        *streak = *streak + 1u;
    } else {
        *streak = 0u;
    }
    *ld_prev = ld;
    *have_prev = true;
    return *streak >= uniforms.interior_windows;
}
