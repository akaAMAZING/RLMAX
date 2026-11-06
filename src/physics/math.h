// RLMAX math layer: the only place with compiler- or target-specific constructs (DESIGN §3.9).
//
// Exact math (the default): IEEE add/mul/div/sqrt, no FMA contraction (nvcc -fmad=false, MSVC /fp:precise), and our
// own polynomial sin/cos/atan2 so the CPU and GPU builds produce identical bits. Fast math (RLMAX_FAST_MATH) swaps in
// hardware intrinsics where they are cheaper; it is not bit-equal across targets.
#pragma once

#include <stdint.h>

// Constant tables of class type (curves, presets): device copies in nvcc's device pass, host constants otherwise.
#if defined(__CUDA_ARCH__)
#define RLMAX_CONST __device__ constexpr
#else
#define RLMAX_CONST constexpr
#endif

// RLMAX_COLD: a rarely taken path kept out of line on the GPU, so it adds no register pressure to the tick
#if defined(__CUDACC__)
#define RLMAX_HD __host__ __device__ __forceinline__
#define RLMAX_D __device__ __forceinline__
#define RLMAX_COLD __host__ __device__ __noinline__
#else
#include <math.h>
#if defined(_MSC_VER)
#define RLMAX_HD __forceinline
#else
#define RLMAX_HD inline __attribute__((always_inline))
#endif
#define RLMAX_D RLMAX_HD
#define RLMAX_COLD inline
#endif

namespace rlmax {

constexpr float PI = 3.14159265358979323846f;
constexpr float FLT_EPS = 1.1920929e-7f;
constexpr float FLT_BIG = 3.4028235e38f;

// ---- scalar ---------------------------------------------------------------------------------------------------------

RLMAX_HD float rsqrt_exact(float x) { return 1.0f / sqrtf(x); }
RLMAX_HD float fmin_(float a, float b) { return a < b ? a : b; }
RLMAX_HD float fmax_(float a, float b) { return a > b ? a : b; }
RLMAX_HD float clampf(float x, float lo, float hi) { return fmin_(fmax_(x, lo), hi); }
RLMAX_HD float absf(float x) { return x < 0.0f ? -x : x; }
// sign with zero -> +1 (Rust f32::signum semantics, ignoring -0 and NaN which never reach here)
RLMAX_HD float signum(float x) { return x < 0.0f ? -1.0f : 1.0f; }
RLMAX_HD float selectf(bool c, float a, float b) { return c ? a : b; }
// Round half to even and round half away from zero, exactly on every target.
RLMAX_HD float round_even(float x) { return rintf(x); }
RLMAX_HD float round_away(float x) { return roundf(x); }

// The game's sinf and cosf (UCRT): a double polynomial with fused steps, rounded to float. Its UCRT port,
// whose matches this machine's ucrtbase.dll on the integrator's arguments where the float polynomial misses 0.1%.
RLMAX_HD double ucrt_sin_poly(double r) {
    const double r2 = r * r;
    double p = fma(r2, 0x1.71de3a556c734p-19, -0x1.a01a01a01a01ap-13);
    p = fma(p, r2, 0x1.1111111111111p-7);
    p = fma(p, r2, -0x1.5555555555555p-3);
    return fma(p, r * r2, r);
}
RLMAX_HD double ucrt_cos_poly(double r) {
    const double r2 = r * r;
    const double base = 1.0 - r2 * 0.5;
    double p = fma(r2, -0x1.27e4fb7789f5cp-22, 0x1.a01a01a01a019p-16);
    p = fma(p, r2, -0x1.6c16c16c16c16p-10);
    p = fma(p, r2, 0x1.5555555555555p-5);
    return fma(p, r2 * r2, base);
}
RLMAX_HD double ucrt_reduce(double a, int* quadrant) {
    const int n = (int)fma(a, 0x1.45f306dc9c883p-1, 0.5);
    const double nd = (double)n;
    const double r1 = fma(-nd, 0x1.921fb54400000p+0, a);
    *quadrant = n & 3;
    return r1 - nd * 0x1.0b4611a626331p-34;
}
RLMAX_HD float ucrt_sinf(float x) {
    const float ax_ = x < 0.0f ? -x : x;
    if (ax_ <= 0.785398185f) {
        // Below this exact cube-boundary, the UCRT cubic rounds back to x (scripts/prove_sin_identity.py).
        if (ax_ < 0x1.d12ed2p-12f) return x;
        if (ax_ < 0x1p-7f) {
            // Compensated binary32 cubic. A narrow midpoint guard retains the UCRT double evaluation only where
            // rounding could be ambiguous; scripts/prove_sin_small.cpp exhausts every input in this branch.
            const float p = ax_ * ax_, pl = fmaf(ax_, ax_, -p), q = p * ax_;
            const float ql = fmaf(p, ax_, -q) + pl * ax_;
            const float hi = fmaf(-q, 0x1.555556p-3f, ax_);
            const float residual = fmaf(-q, 0x1.555556p-3f, ax_ - hi);
            const float correction = fmaf(-ql, 0x1.555556p-3f, fmaf(-q, -0x1.555556p-28f, residual));
            union { float f; uint32_t u; } next;
            next.f = hi;
            next.u += correction < 0.0f ? (uint32_t)-1 : 1u;
            const float gap = fabsf(hi - next.f);
            if (fabsf(fabsf(correction) * 2.0f - gap) <= ax_ * 0x1p-39f) {
                const double d = (double)x;
                return (float)fma(-((d * d) * d), 0x1.5555555555555p-3, d);
            }
            const float result = hi + correction;
            return x < 0.0f ? -result : result;
        }
        if (ax_ < 0x1p-4f) {
            // Compensated fifth-order expansion, with the UCRT polynomial retained at uncertain midpoints.
            // The complete binary32 interval is checked by scripts/prove_sin_medium.cpp.
            const float p = ax_ * ax_, pl = fmaf(ax_, ax_, -p), q = p * ax_;
            const float ql = fmaf(p, ax_, -q) + pl * ax_;
            const float cubic = fmaf(-q, 0x1.555556p-3f, ax_);
            const float residual = fmaf(-q, 0x1.555556p-3f, ax_ - cubic);
            float correction = fmaf(-ql, 0x1.555556p-3f, fmaf(-q, -0x1.555556p-28f, residual));
            const float fifth = (p * p) * ax_;
            const float hi = fmaf(fifth, 0x1.111112p-7f, cubic);
            correction += fmaf(fifth, 0x1.111112p-7f, cubic - hi);
            union { float f; uint32_t u; } next;
            next.f = hi;
            next.u += correction < 0.0f ? (uint32_t)-1 : 1u;
            const float gap = fabsf(hi - next.f);
            if (fabsf(fabsf(correction) * 2.0f - gap) > ax_ * 0x1p-33f) {
                const float result = hi + correction;
                return x < 0.0f ? -result : result;
            }
        } else if (ax_ < 0x1p-2f) {
            // Compensated seventh-order expansion; retain UCRT near a rounding midpoint.
            // scripts/prove_sin_wide.cpp checks both signs of every input in this interval.
            const float p = ax_ * ax_, pl = fmaf(ax_, ax_, -p), q = p * ax_;
            const float ql = fmaf(p, ax_, -q) + pl * ax_;
            const float cubic = fmaf(-q, 0x1.555556p-3f, ax_);
            const float residual = fmaf(-q, 0x1.555556p-3f, ax_ - cubic);
            float correction = fmaf(-ql, 0x1.555556p-3f, fmaf(-q, -0x1.555556p-28f, residual));
            const float p2 = p * p, fifth = p2 * ax_;
            const float h5 = fmaf(fifth, 0x1.111112p-7f, cubic);
            correction += fmaf(fifth, 0x1.111112p-7f, cubic - h5);
            const float seventh = p2 * q, hi = fmaf(-seventh, 0x1.a01a02p-13f, h5);
            correction += fmaf(-seventh, 0x1.a01a02p-13f, h5 - hi);
            union { float f; uint32_t u; } next;
            next.f = hi;
            next.u += correction < 0.0f ? (uint32_t)-1 : 1u;
            const float gap = fabsf(hi - next.f);
            if (fabsf(fabsf(correction) * 2.0f - gap) > ax_ * 0x1p-32f) {
                const float result = hi + correction;
                return x < 0.0f ? -result : result;
            }
        }
        return (float)ucrt_sin_poly((double)x);
    }
    const double d = (double)x;
    int q = 0;
    const double r = ucrt_reduce(d < 0.0 ? -d : d, &q);
    const double y = (q & 1) ? ucrt_cos_poly(r) : ucrt_sin_poly(r);
    return (float)(((q >= 2) != (x < 0.0f)) ? -y : y);
}
RLMAX_HD float ucrt_cosf(float x) {
    const float ax_ = x < 0.0f ? -x : x;
    if (ax_ <= 0.785398185f) {
        if (ax_ < 0x1p-13f) return 1.0f;
        // The binary32 FMA rounds exactly as the double UCRT expression on this range (scripts/prove_cos_small.cpp).
        if (ax_ < 0x1p-7f) return fmaf(-(x * 0.5f), x, 1.0f);
        if (ax_ < 0x1p-4f) {
            // Compensated fourth-order expansion; retain the UCRT polynomial near rounding midpoints.
            // scripts/prove_cos_medium.cpp checks every binary32 input in this interval.
            const float h = x * 0.5f, base = fmaf(-h, x, 1.0f);
            const float residual = fmaf(-h, x, 1.0f - base);
            const float p = x * x, fourth = (p * p) * 0x1.555556p-5f;
            const float hi = base + fourth, correction = ((base - hi) + fourth) + residual;
            // hi and both neighbours lie strictly inside [0.5, 1), so their spacing is 2^-24.
            if (fabsf(fabsf(correction) * 2.0f - 0x1p-24f) > 0x1p-32f) return hi + correction;
        } else if (ax_ < 0x1p-2f) {
            // Compensated eighth-order expansion; retain UCRT near a rounding midpoint.
            // scripts/prove_cos_wide.cpp checks both signs of every input in this interval.
            const float h = x * 0.5f, base = fmaf(-h, x, 1.0f);
            const float residual = fmaf(-h, x, 1.0f - base);
            const float p = x * x, p2 = p * p, fourth = p2 * 0x1.555556p-5f;
            const float h4 = base + fourth;
            float correction = ((base - h4) + fourth) + residual;
            const float sixth = p2 * p, h6 = fmaf(-sixth, 0x1.6c16c2p-10f, h4);
            correction += fmaf(-sixth, 0x1.6c16c2p-10f, h4 - h6);
            const float eighth = p2 * p2, hi = fmaf(eighth, 0x1.a01a02p-16f, h6);
            correction += fmaf(eighth, 0x1.a01a02p-16f, h6 - hi);
            // The same binade bound gives the exact spacing in this interval.
            if (fabsf(fabsf(correction) * 2.0f - 0x1p-24f) > 0x1p-32f) return hi + correction;
        }
        return (float)ucrt_cos_poly((double)x);
    }
    const double d = (double)x;
    int q = 0;
    const double r = ucrt_reduce(d < 0.0 ? -d : d, &q);
    const double y = (q & 1) ? ucrt_sin_poly(r) : ucrt_cos_poly(r);
    return (float)((q == 1 || q == 2) ? -y : y);
}

// The game's atan2f (UCRT): double atan2 rounded to float equals it on 100,000 of 100,000 sampled arguments (SP/rg35/ucrt_cmp.py).
RLMAX_HD float game_atan2f(float y, float x) { return (float)atan2((double)y, (double)x); }

// sin/cos on [-pi/4, pi/4] after a three-part Cody-Waite reduction by pi/2; minimax polynomials (Cephes sinf/cosf
// coefficients). Max error ~1 ulp over |x| < 1e4, far below the game's own noise.
RLMAX_HD void sincos_poly(float x, float* s, float* c) {
#if defined(RLMAX_FAST_MATH) && defined(__CUDA_ARCH__)
    __sincosf(x, s, c);
#else
    const float FOPI = 1.27323954473516f;  // 4/pi
    float ax = absf(x);
    int j = (int)(ax * FOPI);
    j += j & 1;  // round up to even: octant pairs
    float y = (float)j;
    // pi/4 split into three parts so (y * DP) is exact in float.
    const float DP1 = 0.78515625f, DP2 = 2.4187564849853515625e-4f, DP3 = 3.77489497744594108e-8f;
    float z = ((ax - y * DP1) - y * DP2) - y * DP3;
    float zz = z * z;
    float sp = ((-1.9515295891e-4f * zz + 8.3321608736e-3f) * zz - 1.6666654611e-1f) * zz * z + z;
    float cp = ((2.443315711809948e-5f * zz - 1.388731625493765e-3f) * zz + 4.166664568298827e-2f) * zz * zz -
               0.5f * zz + 1.0f;
    int q = (j >> 1) & 3;  // quadrant
    float sv = (q & 1) ? cp : sp;
    float cv = (q & 1) ? sp : cp;
    if (q == 1 || q == 2) cv = -cv;
    if (q >= 2) sv = -sv;
    if (x < 0.0f) sv = -sv;
    *s = sv;
    *c = cv;
#endif
}

// atan on [0, inf) via the Cephes reduction (tan(3pi/8), tan(pi/8)); ~1 ulp.
RLMAX_HD float atan_pos(float x) {
    float y0, xr;
    if (x > 2.414213562373095f) { y0 = PI * 0.5f; xr = -1.0f / x; }
    else if (x > 0.4142135623730950f) { y0 = PI * 0.25f; xr = (x - 1.0f) / (x + 1.0f); }
    else { y0 = 0.0f; xr = x; }
    float z = xr * xr;
    float p = (((8.05374449538e-2f * z - 1.38776856032e-1f) * z + 1.99777106478e-1f) * z - 3.33329491539e-1f) * z * xr + xr;
    return y0 + p;
}
RLMAX_HD float atan2_(float y, float x) {
    if (x == 0.0f) {
        if (y > 0.0f) return PI * 0.5f;
        if (y < 0.0f) return -PI * 0.5f;
        return 0.0f;
    }
    float a = atan_pos(absf(y) / absf(x));
    if (x < 0.0f) a = PI - a;
    return y < 0.0f ? -a : a;
}

// ---- vectors --------------------------------------------------------------------------------------------------------

struct V3 {
    float x, y, z;
};

RLMAX_HD V3 v3(float x, float y, float z) { return V3{x, y, z}; }
RLMAX_HD V3 v3s(float s) { return V3{s, s, s}; }
RLMAX_HD V3 operator+(V3 a, V3 b) { return V3{a.x + b.x, a.y + b.y, a.z + b.z}; }
RLMAX_HD V3 operator-(V3 a, V3 b) { return V3{a.x - b.x, a.y - b.y, a.z - b.z}; }
RLMAX_HD V3 operator-(V3 a) { return V3{-a.x, -a.y, -a.z}; }
RLMAX_HD V3 operator*(V3 a, float s) { return V3{a.x * s, a.y * s, a.z * s}; }
RLMAX_HD V3 operator*(float s, V3 a) { return V3{a.x * s, a.y * s, a.z * s}; }
RLMAX_HD V3 operator*(V3 a, V3 b) { return V3{a.x * b.x, a.y * b.y, a.z * b.z}; }
RLMAX_HD V3 operator/(V3 a, float s) { return V3{a.x / s, a.y / s, a.z / s}; }
RLMAX_HD V3& operator+=(V3& a, V3 b) { a = a + b; return a; }
RLMAX_HD V3& operator-=(V3& a, V3 b) { a = a - b; return a; }
RLMAX_HD V3& operator*=(V3& a, float s) { a = a * s; return a; }
// glam's dot: (x*x' + y*y') + z*z'
RLMAX_HD float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
RLMAX_HD V3 cross(V3 a, V3 b) { return V3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
RLMAX_HD float len2(V3 a) { return dot(a, a); }
RLMAX_HD float len(V3 a) { return sqrtf(len2(a)); }
RLMAX_HD bool is_zero(V3 a) { return a.x == 0.0f && a.y == 0.0f && a.z == 0.0f; }
// glam normalize_or_zero: zero when the length is not finite-positive.
RLMAX_HD V3 normalize_or_zero(V3 a) {
    float l2 = len2(a);
    if (!(l2 > 0.0f) || !(l2 < FLT_BIG)) return v3s(0.0f);
    return a * (1.0f / sqrtf(l2));
}
RLMAX_HD V3 normalize(V3 a) { return a * (1.0f / len(a)); }
RLMAX_HD V3 vmin(V3 a, V3 b) { return V3{fmin_(a.x, b.x), fmin_(a.y, b.y), fmin_(a.z, b.z)}; }
RLMAX_HD V3 vmax(V3 a, V3 b) { return V3{fmax_(a.x, b.x), fmax_(a.y, b.y), fmax_(a.z, b.z)}; }
RLMAX_HD V3 vabs(V3 a) { return V3{absf(a.x), absf(a.y), absf(a.z)}; }
RLMAX_HD V3 vclamp(V3 a, V3 lo, V3 hi) { return vmin(vmax(a, lo), hi); }
RLMAX_HD V3 vsel(bool c, V3 a, V3 b) { return c ? a : b; }
RLMAX_HD float vget(V3 a, int i) { return i == 0 ? a.x : (i == 1 ? a.y : a.z); }
RLMAX_HD V3 vset(V3 a, int i, float v) {
    if (i == 0) a.x = v; else if (i == 1) a.y = v; else a.z = v;
    return a;
}

// ---- matrices (column-major: c0, c1, c2 are the basis axes) -------------------------------------------------------

struct M3 {
    V3 c0, c1, c2;
};

RLMAX_HD M3 m3_identity() { return M3{v3(1, 0, 0), v3(0, 1, 0), v3(0, 0, 1)}; }
RLMAX_HD V3 operator*(const M3& m, V3 v) { return (m.c0 * v.x + m.c1 * v.y) + m.c2 * v.z; }
// transpose(m) * v
RLMAX_HD V3 mul_t(const M3& m, V3 v) { return V3{dot(m.c0, v), dot(m.c1, v), dot(m.c2, v)}; }
RLMAX_HD M3 transpose(const M3& m) {
    return M3{v3(m.c0.x, m.c1.x, m.c2.x), v3(m.c0.y, m.c1.y, m.c2.y), v3(m.c0.z, m.c1.z, m.c2.z)};
}
RLMAX_HD M3 operator*(const M3& a, const M3& b) { return M3{a * b.c0, a * b.c1, a * b.c2}; }

// ---- quaternions (x, y, z, w) ----------------------------------------------------------------------------------------

struct Q {
    float x, y, z, w;
};

RLMAX_HD Q q_identity() { return Q{0, 0, 0, 1}; }
RLMAX_HD Q operator*(Q a, Q b) {
    return Q{a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
             a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
RLMAX_HD Q q_normalize(Q q) {
    float inv = 1.0f / sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return Q{q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}
RLMAX_HD Q q_axis_angle(V3 axis, float angle) {
    float s, c;
    sincos_poly(angle * 0.5f, &s, &c);
    return Q{axis.x * s, axis.y * s, axis.z * s, c};
}
RLMAX_HD M3 q_to_m3(Q q) {
    float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
    float xx = q.x * x2, xy = q.x * y2, xz = q.x * z2;
    float yy = q.y * y2, yz = q.y * z2, zz = q.z * z2;
    float wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;
    return M3{v3(1.0f - (yy + zz), xy + wz, xz - wy), v3(xy - wz, 1.0f - (xx + zz), yz + wx),
              v3(xz + wy, yz - wx, 1.0f - (xx + yy))};
}
RLMAX_HD V3 q_rotate(Q q, V3 v) { return q_to_m3(q) * v; }
// Rotation matrix to quaternion (Shepperd's method, as glam's from_mat3).
RLMAX_HD Q q_from_m3(const M3& m) {
    float m00 = m.c0.x, m01 = m.c0.y, m02 = m.c0.z;
    float m10 = m.c1.x, m11 = m.c1.y, m12 = m.c1.z;
    float m20 = m.c2.x, m21 = m.c2.y, m22 = m.c2.z;
    if (m22 <= 0.0f) {
        float dif10 = m11 - m00, omm22 = 1.0f - m22;
        if (dif10 <= 0.0f) {
            float four_xsq = omm22 - dif10, inv4x = 0.5f / sqrtf(four_xsq);
            return Q{four_xsq * inv4x, (m01 + m10) * inv4x, (m02 + m20) * inv4x, (m12 - m21) * inv4x};
        }
        float four_ysq = omm22 + dif10, inv4y = 0.5f / sqrtf(four_ysq);
        return Q{(m01 + m10) * inv4y, four_ysq * inv4y, (m12 + m21) * inv4y, (m20 - m02) * inv4y};
    }
    float sum10 = m11 + m00, opm22 = 1.0f + m22;
    if (sum10 <= 0.0f) {
        float four_zsq = opm22 - sum10, inv4z = 0.5f / sqrtf(four_zsq);
        return Q{(m02 + m20) * inv4z, (m12 + m21) * inv4z, four_zsq * inv4z, (m01 - m10) * inv4z};
    }
    float four_wsq = opm22 + sum10, inv4w = 0.5f / sqrtf(four_wsq);
    return Q{(m12 - m21) * inv4w, (m20 - m02) * inv4w, (m01 - m10) * inv4w, four_wsq * inv4w};
}

}  // namespace rlmax
