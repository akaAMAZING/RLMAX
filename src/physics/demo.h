// Demolition's swept hitboxes and rotator-space cone. These run only for car-car callbacks.
#pragma once
#include "car_tick.h"

namespace rlmax {

struct CarPose {
    V3 pos;
    M3 r;
    Q q;
    V3 uu;
};

// UCRT acos, used by the game's rotator and cone calculations. Inputs here are clamped to [-1, 1].
RLMAX_COLD float demo_acos(float x) {
    const float a = absf(x);
    if (a < 0x1p-26f) return 0x1.921fb6p+0f;
    if (a >= 1.0f) return x > 0.0f ? 0.0f : 0x1.921fb6p+1f;
    const bool big = a >= 0.5f;
    const float z = big ? (1.0f - a) * 0.5f : a * a;
    const float s = big ? sqrtf(z) : 0.0f;
    float p = 0x1.039cd6p-8f;
    p = fmaf(-p, z, -0x1.b67fc2p-7f);
    p = fmaf(p, z, -0x1.cf17bap-5f);
    p = fmaf(p, z, 0x1.7929b8p-3f);
    const float r = (p * z) / fmaf(-0x1.ac3e1ap-1f, z, 0x1.1adf4ap+0f);
    if (!big) {
        const double lo = 0x1.1a62633145c07p-54 - (double)(r * x);
        return (float)(0x1.921fb54442d18p+0 - ((double)x - lo));
    }
    if (x < 0.0f) {
        const double h = ((double)(s * r) - 0x1.1a62633145c07p-54) + (double)s;
        return (float)(0x1.921fb54442d18p+1 - (h + h));
    }
    union {
        float f;
        uint32_t u;
    } b{s};
    b.u &= 0xffff0000u;
    const float c = fmaf(-b.f, b.f, z) / (s + b.f);
    return fmaf(r, s + s, c + c) + (b.f + b.f);
}

RLMAX_HD V3 demo_unit(V3 v) {
    const float l2 = len2(v);
    if (l2 == 1.0f) return v;
    return l2 < 1e-8f ? v3s(0.0f) : v * (1.0f / sqrtf(l2));
}

RLMAX_COLD bool within_forward_cone(const CarPose &a, V3 aoff, const CarPose &v, V3 voff, bool reverse, float max_yaw,
                                    float max_pitch) {
    const M3 ar = ue3_rotator_axes(a.q), vr = ue3_rotator_axes(v.q);
    const V3 f = reverse ? -ar.c0 : ar.c0;
    const V3 d = demo_unit((v.uu + mul_t(vr, voff)) - (a.uu + mul_t(ar, aoff)));
    auto angle = [&](V3 axis) {
        float k = dot(d, axis);
        k = k >= 0.0f ? k / 1.01f : k * 1.01f;
        return demo_acos(clampf(dot(f, demo_unit(d - axis * k)), -1.0f, 1.0f)) * 57.29578f;
    };
    return angle(ar.c1) <= max_pitch && angle(ar.c2) <= max_yaw;
}

// Relative angular displacement between two sweep endpoint bases (Bullet calculateVelocity).
RLMAX_COLD V3 demo_angular_displacement(const M3 &a, const M3 &b) {
    const float m00 = a.c0.x, m01 = a.c1.x, m02 = a.c2.x, m10 = a.c0.y, m11 = a.c1.y, m12 = a.c2.y, m20 = a.c0.z,
                m21 = a.c1.z, m22 = a.c2.z;
    const float c0 = m11 * m22 - m12 * m21, c1 = m12 * m20 - m10 * m22, c2 = m10 * m21 - m11 * m20;
    const float s = 1.0f / ((m00 * c0 + m01 * c1) + m02 * c2);
    const float i00 = c0 * s, i01 = (m02 * m21 - m01 * m22) * s, i02 = (m01 * m12 - m02 * m11) * s;
    const float i10 = c1 * s, i11 = (m00 * m22 - m02 * m20) * s, i12 = (m02 * m10 - m00 * m12) * s;
    const float i20 = c2 * s, i21 = (m01 * m20 - m00 * m21) * s, i22 = (m00 * m11 - m01 * m10) * s;
    auto row = [&](float x, float y, float z) {
        return v3((i00 * x + i10 * y) + i20 * z, (i01 * x + i11 * y) + i21 * z, (i02 * x + i12 * y) + i22 * z);
    };
    const V3 r0 = row(b.c0.x, b.c1.x, b.c2.x), r1 = row(b.c0.y, b.c1.y, b.c2.y), r2 = row(b.c0.z, b.c1.z, b.c2.z);
    Q q = q_normalize(bullet_get_rotation(M3{v3(r0.x, r1.x, r2.x), v3(r0.y, r1.y, r2.y), v3(r0.z, r1.z, r2.z)}));
    const float angle = 2.0f * demo_acos(clampf(q.w, -1.0f, 1.0f));
    const float l2 = (q.x * q.x + q.y * q.y) + q.z * q.z;
    V3 axis = v3(1.0f, 0.0f, 0.0f);
    if (l2 >= 1.42108547e-14f) axis = v3(q.x, q.y, q.z) * (1.0f / sqrtf(l2));
    return axis * angle;
}

RLMAX_COLD CarPose demo_integrate(CarPose p, V3 lin, V3 ang, float dt) {
    Body b{};
    b.pos = p.pos;
    b.r = p.r;
    integrate_game(b, lin, ang, dt);
    return CarPose{b.pos, b.r, b.rot, p.uu};
}

RLMAX_COLD Q demo_slerp(Q a, Q b, float t) {
    const float raw = ((a.x * b.x + a.y * b.y) + a.z * b.z) + a.w * b.w;
    const float c = absf(raw);
    float x = 1.0f - t, y = t;
    if (c < 0.9999f) {
        const float omega = demo_acos(clampf(c, -1.0f, 1.0f)), inv = 1.0f / ucrt_sinf(omega);
        x = ucrt_sinf((1.0f - t) * omega) * inv;
        y = ucrt_sinf(t * omega) * inv;
    }
    if (raw < 0.0f) y = -y;
    Q q{y * b.x + a.x * x, y * b.y + a.y * x, y * b.z + a.z * x, y * b.w + a.w * x};
    const float n = ((q.x * q.x + q.y * q.y) + q.z * q.z) + q.w * q.w;
    if (!(n > 1e-8f)) return Q{0, 0, 0, 1};
    const float inv = 1.0f / sqrtf(n);
    return Q{q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

// GJK distance between the boxes' implicit cores, then their margins and the penetration solver if needed.
RLMAX_COLD bool demo_box_distance(const CarPose &a, const CarGeo &ag, const CarPose &b, const CarGeo &bg,
                                  EpaScratch &scratch, float &distance, V3 &normal) {
    const V3 shift = (a.pos + b.pos) * 0.5f, pa0 = a.pos - shift, pb0 = b.pos - shift;
    GjkSimplex s;
    gjk_reset(s);
    V3 axis = v3(0, 1, 0);
    float sq = 1e30f;
    int deg = 0, iter = 0;
    bool check = false;
    for (;;) {
        const V3 pa = a.r * gjk_box_support(ag.hitbox_half, mul_t(a.r, -axis)) + pa0;
        const V3 pb = b.r * gjk_box_support(bg.hitbox_half, mul_t(b.r, axis)) + pb0;
        const V3 w = pa - pb;
        const float delta = dot(axis, w);
        if (delta > 0.0f && delta * delta > sq * 1e18f) {
            deg = 10;
            check = true;
            break;
        }
        if (gjk_in_simplex(s, w)) {
            deg = 1;
            check = true;
            break;
        }
        const float f0 = sq - delta;
        if (f0 <= sq * GJK_REL_ERROR2) {
            deg = f0 <= 0.0f ? 2 : 11;
            check = true;
            break;
        }
        gjk_add(s, w, pa, pb);
        if (!gjk_update(s)) {
            deg = 3;
            check = true;
            break;
        }
        const V3 next = s.cv;
        if (len2(next) < GJK_REL_ERROR2) {
            axis = next;
            deg = 6;
            check = true;
            break;
        }
        const float old = sq;
        sq = len2(next);
        if (old - sq <= FLT_EPS * old) {
            deg = 12;
            check = true;
            break;
        }
        axis = next;
        if (iter++ > GJK_MAX_ITER || s.n == 4) break;
    }
    if (!check) return false;
    gjk_update(s);
    const float l2 = len2(axis), margins = ag.margin + bg.margin;
    if (l2 <= 0.0001f) deg = 5;
    bool valid = false;
    if (l2 > FLT_EPS * FLT_EPS && sqrtf(sq) > 0.0f) {
        const float inv = 1.0f / sqrtf(l2);
        normal = axis * inv;
        distance = 1.0f / inv - margins;
        valid = true;
    }
    if (!valid || (deg && distance + margins <= 0.01f)) {
        const PenResult p =
            epa_box_box(ag.hitbox_half, ag.margin, a.r, pa0, bg.hitbox_half, bg.margin, b.r, pb0, scratch);
        if (p.pen) {
            V3 n = p.wb - p.wa;
            float n2 = len2(n);
            if (n2 <= FLT_EPS * FLT_EPS) {
                n = p.v;
                n2 = len2(n);
            }
            if (n2 > FLT_EPS * FLT_EPS) {
                const float d = -len(p.wa - p.wb);
                if (!valid || d < distance) {
                    distance = d;
                    normal = n * (1.0f / sqrtf(n2));
                    valid = true;
                }
            }
        } else if (p.dist && len2(p.v) > 0.0f) {
            const float d = len(p.wa - p.wb) - margins;
            if (!valid || d < distance) {
                distance = d;
                normal = demo_unit(p.v);
                valid = true;
            }
        }
    }
    if (!valid) return false;
    const V3 ea = obb_extent(Obb{pa0, a.r, ag.hitbox_half, ag.margin});
    const V3 eb = obb_extent(Obb{pb0, b.r, bg.hitbox_half, bg.margin});
    const V3 ca = ((pa0 + ea) + (pa0 - ea)) * 0.5f, cb = ((pb0 + eb) + (pb0 - eb)) * 0.5f;
    if (dot(ca - cb, normal) < 0.0f) normal = -normal;
    return true;
}

// Conservative advancement of both child boxes, with the game's angular-motion bound and allowed penetration.
RLMAX_COLD void time_of_impact_poses(CarPose &a, const CarGeo &ag, V3 av, V3 aw, CarPose &b, const CarGeo &bg, V3 bv,
                                     V3 bw, EpaScratch &scratch) {
    auto endpoints = [](const CarPose &p, V3 v, V3 w, CarPose &from, CarPose &to) {
        CarPose t{p.uu * c::UU_TO_BT, bullet_set_rotation(p.q), p.q, p.uu};
        const V3 lin = v * c::UU_TO_BT;
        from = demo_integrate(t, -lin, -w, c::DT);
        to = demo_integrate(t, lin, w, c::DT);
    };
    CarPose af, at, bf, bt;
    endpoints(a, av, aw, af, at);
    endpoints(b, bv, bw, bf, bt);
    auto child = [](CarPose p, const CarGeo &g) {
        p.pos = p.pos + p.r * g.hitbox_off;
        return p;
    };
    const CarPose a0 = child(af, ag), a1 = child(at, ag), b0 = child(bf, bg), b1 = child(bt, bg);
    const V3 la = a1.pos - a0.pos, lb = b1.pos - b0.pos, aa = demo_angular_displacement(a0.r, a1.r),
             ab = demo_angular_displacement(b0.r, b1.r);
    const float bound = len(aa) * len(ag.hitbox_half + v3s(ag.margin)) + len(ab) * len(bg.hitbox_half + v3s(bg.margin));
    const V3 rel = lb - la;
    if (len(rel) + bound == 0.0f) return;
    float d = 0.0f;
    V3 n;
    if (!demo_box_distance(a0, ag, b0, bg, scratch, d, n) || dot(rel, n) + bound <= FLT_EPS) return;
    float dist = d + 0.04f, lambda = 0.0f, fraction = 0.0f;
    int iter = 0;
    while (dist > 0.001f) {
        const float proj = dot(rel, n) + bound;
        if (proj <= FLT_EPS) return;
        fraction = lambda + dist / proj;
        if (fraction > 1.0f || fraction < 0.0f || fraction <= lambda) return;
        const CarPose ai = demo_integrate(a0, la, aa, fraction), bi = demo_integrate(b0, lb, ab, fraction);
        if (!demo_box_distance(ai, ag, bi, bg, scratch, d, n)) return;
        dist = d + 0.04f;
        if (++iter > 64) return;
        lambda = fraction;
    }
    if (!(fraction < 1.0f)) return;
    auto pose = [&](CarPose &p, const CarPose &from, const CarPose &to) {
        const V3 x = from.pos * c::BT_TO_UU, y = to.pos * c::BT_TO_UU;
        p.uu = (y - x) * fraction + x;
        p.pos = p.uu * c::UU_TO_BT;
        p.q = demo_slerp(bullet_get_rotation(from.r), bullet_get_rotation(to.r), fraction);
        p.r = bullet_set_rotation(p.q);
    };
    pose(a, af, at);
    pose(b, bf, bt);
}
} // namespace rlmax
