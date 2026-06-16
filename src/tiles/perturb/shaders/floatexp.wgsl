// Shared-exponent complex floatexp.
//
// A value is  m · 2^e  where `m` is a complex (vec2) mantissa and `e` an
// integer exponent shared by both components.  This keeps f32's full 24-bit
// mantissa while extending the exponent range from ~±127 to ~±2e9, so the
// per-pixel delta iteration survives zoom levels where plain f32 underflows.
//
// The mantissa is kept normalized so its larger component is in [0.5, 1).

struct Fe {
    m : vec2<f32>,   // complex mantissa (x = re, y = im)
    e : i32,         // shared power-of-two exponent
}

// Sentinel exponent for the value zero.
const FE_ZERO_EXP : i32 = -2000000000;

// Renormalize so max(|m.x|, |m.y|) lands in [0.5, 1).
fn fe_norm(a : Fe) -> Fe {
    let mx = max(abs(a.m.x), abs(a.m.y));
    if (mx == 0.0) {
        return Fe(vec2<f32>(0.0, 0.0), FE_ZERO_EXP);
    }
    let r = frexp(mx);                                  // mx = r.fract · 2^r.exp
    let m = ldexp(a.m, vec2<i32>(-r.exp, -r.exp));
    return Fe(m, a.e + r.exp);
}

// Plain f32 complex -> Fe.
fn fe_from_c(v : vec2<f32>) -> Fe {
    return fe_norm(Fe(v, 0));
}

// Fe -> plain f32 complex; underflows to 0 when far below f32 range.
fn fe_to_c(a : Fe) -> vec2<f32> {
    return ldexp(a.m, vec2<i32>(a.e, a.e));
}

fn fe_add(a : Fe, b : Fe) -> Fe {
    if (a.e >= b.e) {
        let s = b.e - a.e;                              // <= 0
        return fe_norm(Fe(a.m + ldexp(b.m, vec2<i32>(s, s)), a.e));
    }
    let s = a.e - b.e;                                  // < 0
    return fe_norm(Fe(b.m + ldexp(a.m, vec2<i32>(s, s)), b.e));
}

// Complex multiply.
fn fe_mul(a : Fe, b : Fe) -> Fe {
    let m = vec2<f32>(
        a.m.x * b.m.x - a.m.y * b.m.y,
        a.m.x * b.m.y + a.m.y * b.m.x,
    );
    return fe_norm(Fe(m, a.e + b.e));
}
