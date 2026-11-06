// Bullet's distance query for a box against a triangle, as the game's convex-convex algorithm runs it per mesh triangle:
// btGjkPairDetector::getClosestPoints with a btVoronoiSimplexSolver, on the shapes without their margins (support
// points of the implicit box and of the triangle), the margins then taken off the distance and the witnesses pushed
// out along the axis. Written from Bullet's documented algorithm (Johnson's distance subalgorithm on Voronoi regions,
// Ericson 5.1.5 and 5.1.6 for the triangle and tetrahedron cases); the operation order follows Bullet's so that the
// witness in a degenerate (flat) configuration lands where Bullet's does. Fixed storage, bounded iterations.
// The game's detector: bin 2a79c0 (truth/bin/tri_aabb_gate.md).
#pragma once

#include "consts.h"
#include "math.h"

namespace rlmax {

constexpr float GJK_REL_ERROR2 = 1.0e-6f;
constexpr float GJK_EQUAL_VERTEX_THRESHOLD = 1.0e-4f;  // compared against the squared distance, as Bullet does
constexpr float GJK_LARGE = 1.0e18f;                    // BT_LARGE_FLOAT
constexpr int GJK_MAX_ITER = 1000;
// float allowance on the plane-separation reject (contacts.h obb_tri): 1e-3 BT, far above the rounding of either side
constexpr float GJK_PLANE_REJECT_ALLOWANCE = 1.0e-3f;

// btSubSimplexClosestResult
struct GjkBC {
    V3 closest;
    uint32_t used;  // bit i: vertex i used
    float bc[4];
    bool degenerate;
};

RLMAX_HD void gjk_bc_reset(GjkBC& r) {
    r.degenerate = false;
    r.bc[0] = r.bc[1] = r.bc[2] = r.bc[3] = 0.0f;
    r.used = 0;
}
RLMAX_HD void gjk_bc_set(GjkBC& r, float a, float b, float c, float d) {
    r.bc[0] = a;
    r.bc[1] = b;
    r.bc[2] = c;
    r.bc[3] = d;
}
RLMAX_HD bool gjk_bc_valid(const GjkBC& r) {
    return r.bc[0] >= 0.0f && r.bc[1] >= 0.0f && r.bc[2] >= 0.0f && r.bc[3] >= 0.0f;
}

// btVoronoiSimplexSolver
struct GjkSimplex {
    V3 w[4], p[4], q[4];
    int n;
    V3 cp1, cp2, cv, last_w;
    bool cvalid, needs_update;
    GjkBC cbc;
};

RLMAX_HD void gjk_reset(GjkSimplex& s) {
    s.cvalid = false;
    s.n = 0;
    s.needs_update = true;
    s.last_w = v3s(GJK_LARGE);
    gjk_bc_reset(s.cbc);
}

RLMAX_HD void gjk_remove(GjkSimplex& s, int i) {
    s.n--;
    s.w[i] = s.w[s.n];
    s.p[i] = s.p[s.n];
    s.q[i] = s.q[s.n];
}

RLMAX_HD void gjk_reduce(GjkSimplex& s, uint32_t used) {
    if (s.n >= 4 && !(used & 8u)) gjk_remove(s, 3);
    if (s.n >= 3 && !(used & 4u)) gjk_remove(s, 2);
    if (s.n >= 2 && !(used & 2u)) gjk_remove(s, 1);
    if (s.n >= 1 && !(used & 1u)) gjk_remove(s, 0);
}

// closest point of triangle abc to p (Ericson 5.1.5, Bullet's closestPtPointTriangle)
RLMAX_HD void gjk_closest_tri(V3 p, V3 a, V3 b, V3 c, GjkBC& r) {
    r.used = 0;
    V3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        r.closest = a;
        r.used = 1u;
        gjk_bc_set(r, 1, 0, 0, 0);
        return;
    }
    V3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        r.closest = b;
        r.used = 2u;
        gjk_bc_set(r, 0, 1, 0, 0);
        return;
    }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        r.closest = a + v * ab;
        r.used = 1u | 2u;
        gjk_bc_set(r, 1 - v, v, 0, 0);
        return;
    }
    V3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        r.closest = c;
        r.used = 4u;
        gjk_bc_set(r, 0, 0, 1, 0);
        return;
    }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        r.closest = a + w * ac;
        r.used = 1u | 4u;
        gjk_bc_set(r, 1 - w, 0, w, 0);
        return;
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        r.closest = b + w * (c - b);
        r.used = 2u | 4u;
        gjk_bc_set(r, 0, 1 - w, w, 0);
        return;
    }
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom, w = vc * denom;
    r.closest = a + ab * v + ac * w;
    r.used = 1u | 2u | 4u;
    gjk_bc_set(r, 1 - v - w, v, w, 0);
}

// p and d on opposite sides of plane abc: 1, same side: 0, degenerate (d on the plane): -1
RLMAX_HD int gjk_outside_plane(V3 p, V3 a, V3 b, V3 c, V3 d) {
    V3 nrm = cross(b - a, c - a);
    float signp = dot(p - a, nrm), signd = dot(d - a, nrm);
    if (signd * signd < 1.0e-4f * 1.0e-4f) return -1;
    return signp * signd < 0.0f ? 1 : 0;
}

// closest point of tetrahedron abcd to p (Ericson 5.1.6, Bullet's closestPtPointTetrahedron): false when p is inside
// (or the tetrahedron degenerate)
RLMAX_HD bool gjk_closest_tet(V3 p, V3 a, V3 b, V3 c, V3 d, GjkBC& fr) {
    GjkBC t;
    gjk_bc_reset(t);
    fr.closest = p;
    fr.used = 15u;
    int o_abc = gjk_outside_plane(p, a, b, c, d);
    int o_acd = gjk_outside_plane(p, a, c, d, b);
    int o_adb = gjk_outside_plane(p, a, d, b, c);
    int o_bdc = gjk_outside_plane(p, b, d, c, a);
    if (o_abc < 0 || o_acd < 0 || o_adb < 0 || o_bdc < 0) {
        fr.degenerate = true;
        return false;
    }
    if (!o_abc && !o_acd && !o_adb && !o_bdc) return false;
    float best = FLT_BIG;
    auto bit = [](uint32_t u, int i) { return (u >> i) & 1u; };
    if (o_abc) {
        gjk_closest_tri(p, a, b, c, t);
        V3 q = t.closest;
        float sq = dot(q - p, q - p);
        if (sq < best) {
            best = sq;
            fr.closest = q;
            fr.used = bit(t.used, 0) | (bit(t.used, 1) << 1) | (bit(t.used, 2) << 2);
            gjk_bc_set(fr, t.bc[0], t.bc[1], t.bc[2], 0);
        }
    }
    if (o_acd) {
        gjk_closest_tri(p, a, c, d, t);
        V3 q = t.closest;
        float sq = dot(q - p, q - p);
        if (sq < best) {
            best = sq;
            fr.closest = q;
            fr.used = bit(t.used, 0) | (bit(t.used, 1) << 2) | (bit(t.used, 2) << 3);
            gjk_bc_set(fr, t.bc[0], 0, t.bc[1], t.bc[2]);
        }
    }
    if (o_adb) {
        gjk_closest_tri(p, a, d, b, t);
        V3 q = t.closest;
        float sq = dot(q - p, q - p);
        if (sq < best) {
            best = sq;
            fr.closest = q;
            fr.used = bit(t.used, 0) | (bit(t.used, 2) << 1) | (bit(t.used, 1) << 3);
            gjk_bc_set(fr, t.bc[0], t.bc[2], 0, t.bc[1]);
        }
    }
    if (o_bdc) {
        gjk_closest_tri(p, b, d, c, t);
        V3 q = t.closest;
        float sq = dot(q - p, q - p);
        if (sq < best) {
            best = sq;
            fr.closest = q;
            fr.used = (bit(t.used, 0) << 1) | (bit(t.used, 2) << 2) | (bit(t.used, 1) << 3);
            gjk_bc_set(fr, 0, t.bc[0], t.bc[2], t.bc[1]);
        }
    }
    return true;
}

RLMAX_HD bool gjk_update(GjkSimplex& s) {
    if (!s.needs_update) return s.cvalid;
    gjk_bc_reset(s.cbc);
    s.needs_update = false;
    const V3 o = v3s(0.0f);
    switch (s.n) {
        case 0:
            s.cvalid = false;
            break;
        case 1:
            s.cp1 = s.p[0];
            s.cp2 = s.q[0];
            s.cv = s.cp1 - s.cp2;
            gjk_bc_reset(s.cbc);
            gjk_bc_set(s.cbc, 1, 0, 0, 0);
            s.cvalid = gjk_bc_valid(s.cbc);
            break;
        case 2: {
            V3 from = s.w[0], to = s.w[1];
            V3 diff = o - from;
            V3 v = to - from;
            float t = dot(v, diff);
            if (t > 0.0f) {
                float vv = dot(v, v);
                if (t < vv) {
                    t /= vv;
                    diff -= t * v;
                    s.cbc.used = 1u | 2u;
                } else {
                    t = 1.0f;
                    diff -= v;
                    s.cbc.used = 2u;
                }
            } else {
                t = 0.0f;
                s.cbc.used = 1u;
            }
            gjk_bc_set(s.cbc, 1 - t, t, 0, 0);
            s.cp1 = s.p[0] + t * (s.p[1] - s.p[0]);
            s.cp2 = s.q[0] + t * (s.q[1] - s.q[0]);
            s.cv = s.cp1 - s.cp2;
            gjk_reduce(s, s.cbc.used);
            s.cvalid = gjk_bc_valid(s.cbc);
            break;
        }
        case 3: {
            gjk_closest_tri(o, s.w[0], s.w[1], s.w[2], s.cbc);
            s.cp1 = s.p[0] * s.cbc.bc[0] + s.p[1] * s.cbc.bc[1] + s.p[2] * s.cbc.bc[2];
            s.cp2 = s.q[0] * s.cbc.bc[0] + s.q[1] * s.cbc.bc[1] + s.q[2] * s.cbc.bc[2];
            s.cv = s.cp1 - s.cp2;
            gjk_reduce(s, s.cbc.used);
            s.cvalid = gjk_bc_valid(s.cbc);
            break;
        }
        case 4: {
            bool sep = gjk_closest_tet(o, s.w[0], s.w[1], s.w[2], s.w[3], s.cbc);
            if (sep) {
                s.cp1 = s.p[0] * s.cbc.bc[0] + s.p[1] * s.cbc.bc[1] + s.p[2] * s.cbc.bc[2] + s.p[3] * s.cbc.bc[3];
                s.cp2 = s.q[0] * s.cbc.bc[0] + s.q[1] * s.cbc.bc[1] + s.q[2] * s.cbc.bc[2] + s.q[3] * s.cbc.bc[3];
                s.cv = s.cp1 - s.cp2;
                gjk_reduce(s, s.cbc.used);
            } else {
                if (s.cbc.degenerate) {
                    s.cvalid = false;
                } else {
                    s.cvalid = true;
                    s.cv = o;
                }
                break;
            }
            s.cvalid = gjk_bc_valid(s.cbc);
            break;
        }
        default:
            s.cvalid = false;
    }
    return s.cvalid;
}

RLMAX_HD void gjk_add(GjkSimplex& s, V3 w, V3 p, V3 q) {
    s.last_w = w;
    s.needs_update = true;
    s.w[s.n] = w;
    s.p[s.n] = p;
    s.q[s.n] = q;
    s.n++;
}

RLMAX_HD bool gjk_in_simplex(const GjkSimplex& s, V3 w) {
    bool found = false;
    for (int i = 0; i < 4; i++) {
        if (i >= s.n) break;
        if (len2(s.w[i] - w) <= GJK_EQUAL_VERTEX_THRESHOLD) {
            found = true;
            break;
        }
    }
    if (w.x == s.last_w.x && w.y == s.last_w.y && w.z == s.last_w.z) return true;
    return found;
}

// support of the implicit box (btFsels: a component >= 0 takes +h)
RLMAX_HD V3 gjk_box_support(V3 h, V3 d) {
    return v3(d.x >= 0.0f ? h.x : -h.x, d.y >= 0.0f ? h.y : -h.y, d.z >= 0.0f ? h.z : -h.z);
}
// support of a triangle: the vertex of largest dot, btVector3::maxAxis on the three dots
RLMAX_HD V3 gjk_tri_support(const V3 t[3], V3 d) {
    float a = dot(d, t[0]), b = dot(d, t[1]), c = dot(d, t[2]);
    int i = a < b ? (b < c ? 2 : 1) : (a < c ? 2 : 0);
    return t[i];
}

// ---- btGjkEpa2: its own GJK (with margins) to enclose the origin, then EPA on the polytope ----------------------------
// Bullet 2.88's btGjkEpaSolver2 at the game's sizes and constants (bin 2a0ca0, 14029f6c0, 29bed0;
// truth/bin/tri_aabb_gate.md). Shape 0 is the box in its own frame, its support with margin the implicit vertex plus
// margin times the unit direction (a rounded box, btConvexShape::localGetSupportVertexNonVirtual); shape 1 is the
// triangle (margin 0), carried into the box frame. Bullet divides a vector by a scalar as a multiply by the
// reciprocal, and so does this code (bdiv).

constexpr int EPA2_GJK_MAX_ITER = 128;
constexpr float EPA2_GJK_ACCURACY = 1.0e-4f, EPA2_GJK_MIN_DISTANCE = 1.0e-4f, EPA2_GJK_DUPLICATED_EPS = 1.0e-4f;
constexpr int EPA_MAX_VERTICES = 64, EPA_MAX_FACES = 128, EPA_MAX_ITERATIONS = 255;
constexpr float EPA_ACCURACY = 1.0e-4f, EPA_PLANE_EPS = 1.0e-5f;

RLMAX_HD V3 bdiv(V3 v, float s) { return v * (1.0f / s); }

struct Epa2Shape {
    V3 h;  // implicit half extents of the box
    float margin;
    V3 h2;  // second box, used only by the box-box cast
    float margin2;
    bool with_margin;
    M3 to1;    // box frame -> triangle frame (basis)
    M3 to0_b;  // triangle frame -> box frame (basis)
    V3 to0_o;  // ... origin
    V3 tv[3];  // triangle vertices, triangle frame
};

RLMAX_HD V3 epa2_support0(const Epa2Shape& s, V3 d) {
    if (!s.with_margin) return gjk_box_support(s.h, d);
    V3 n = d;
    if (len2(n) < FLT_EPS * FLT_EPS) n = v3s(-1.0f);
    n = bdiv(n, len(n));
    return gjk_box_support(s.h, n) + n * s.margin;
}
template <bool BOX = false>
RLMAX_HD V3 epa2_support1(const Epa2Shape& s, V3 d) {
    V3 dir = s.to1 * d;
    // The with-margin support normalizes before selecting a vertex even when the triangle's margin is zero. Rounding
    // the direction can change which vertex wins a near tie (Bullet localGetSupportVertexNonVirtual).
    if (s.with_margin) {
        if (len2(dir) < FLT_EPS * FLT_EPS) dir = v3s(-1.0f);
        dir = bdiv(dir, len(dir));
    }
    V3 t;
    if constexpr (BOX) t = gjk_box_support(s.h2, dir);
    else t = gjk_tri_support(s.tv, dir);
    if (s.with_margin) t += dir * (BOX ? s.margin2 : 0.0f);
    return s.to0_b * t + s.to0_o;
}
template <bool BOX = false>
RLMAX_HD V3 epa2_support(const Epa2Shape& s, V3 d) { return epa2_support0(s, d) - epa2_support1<BOX>(s, -d); }

struct Epa2SV {
    V3 d, w;
};
struct Epa2Simplex {
    int c[4];
    float p[4];
    int rank;
};

// vertex pool: 0..3 the GJK store, 4.. the EPA store
struct Epa2 {
    Epa2Shape shape;
    Epa2SV sv[4 + EPA_MAX_VERTICES];
    int free_[4];
    int nfree;
    V3 ray;
    float distance;
    Epa2Simplex simplices[2];
    int current;
    int simplex;  // index into simplices
    int status;   // 0 valid, 1 inside, 2 failed
};

template <bool BOX = false>
RLMAX_HD void epa2_getsupport(Epa2& g, V3 d, Epa2SV& v) {
    v.d = bdiv(d, len(d));
    v.w = epa2_support<BOX>(g.shape, v.d);
}
RLMAX_HD void epa2_remove(Epa2& g, Epa2Simplex& s) { g.free_[g.nfree++] = s.c[--s.rank]; }
template <bool BOX = false>
RLMAX_HD void epa2_append(Epa2& g, Epa2Simplex& s, V3 v) {
    s.p[s.rank] = 0.0f;
    s.c[s.rank] = g.free_[--g.nfree];
    epa2_getsupport<BOX>(g, v, g.sv[s.c[s.rank++]]);
}
RLMAX_HD float epa2_det(V3 a, V3 b, V3 c) {
    return a.y * b.z * c.x + a.z * b.x * c.y - a.x * b.z * c.y - a.y * b.x * c.z + a.x * b.y * c.z - a.z * b.y * c.x;
}
RLMAX_HD float epa2_proj2(V3 a, V3 b, float* w, uint32_t& m) {
    const V3 d = b - a;
    const float l = len2(d);
    if (l > 0.0f) {
        const float t = l > 0.0f ? -dot(a, d) / l : 0.0f;
        if (t >= 1.0f) {
            w[0] = 0.0f;
            w[1] = 1.0f;
            m = 2;
            return len2(b);
        } else if (t <= 0.0f) {
            w[0] = 1.0f;
            w[1] = 0.0f;
            m = 1;
            return len2(a);
        } else {
            w[1] = t;
            w[0] = 1.0f - t;
            m = 3;
            return len2(a + d * t);
        }
    }
    return -1.0f;
}
RLMAX_HD float epa2_proj3(V3 a, V3 b, V3 c, float* w, uint32_t& m) {
    const uint32_t imd3[3] = {1, 2, 0};
    const V3 vt[3] = {a, b, c};
    const V3 dl[3] = {a - b, b - c, c - a};
    const V3 n = cross(dl[0], dl[1]);
    const float l = len2(n);
    if (l > 0.0f) {
        float mindist = -1.0f;
        float subw[2] = {0.0f, 0.0f};
        uint32_t subm = 0;
        for (uint32_t i = 0; i < 3; ++i) {
            if (dot(vt[i], cross(dl[i], n)) > 0.0f) {
                const uint32_t j = imd3[i];
                const float subd = epa2_proj2(vt[i], vt[j], subw, subm);
                if (mindist < 0.0f || subd < mindist) {
                    mindist = subd;
                    m = ((subm & 1u) ? 1u << i : 0u) + ((subm & 2u) ? 1u << j : 0u);
                    w[i] = subw[0];
                    w[j] = subw[1];
                    w[imd3[j]] = 0.0f;
                }
            }
        }
        if (mindist < 0.0f) {
            const float d = dot(a, n);
            const float s = sqrtf(l);
            const V3 p = n * (d / l);
            mindist = len2(p);
            m = 7;
            w[0] = len(cross(dl[1], b - p)) / s;
            w[1] = len(cross(dl[2], c - p)) / s;
            w[2] = 1.0f - (w[0] + w[1]);
        }
        return mindist;
    }
    return -1.0f;
}
RLMAX_HD float epa2_proj4(V3 a, V3 b, V3 c, V3 d, float* w, uint32_t& m) {
    const uint32_t imd3[3] = {1, 2, 0};
    const V3 vt[4] = {a, b, c, d};
    const V3 dl[3] = {a - d, b - d, c - d};
    const float vl = epa2_det(dl[0], dl[1], dl[2]);
    const bool ng = (vl * dot(a, cross(b - c, a - b))) <= 0.0f;
    if (ng && absf(vl) > 0.0f) {
        float mindist = -1.0f;
        float subw[3] = {0.0f, 0.0f, 0.0f};
        uint32_t subm = 0;
        for (uint32_t i = 0; i < 3; ++i) {
            const uint32_t j = imd3[i];
            const float s = vl * dot(d, cross(dl[i], dl[j]));
            if (s > 0.0f) {
                const float subd = epa2_proj3(vt[i], vt[j], d, subw, subm);
                if (mindist < 0.0f || subd < mindist) {
                    mindist = subd;
                    m = ((subm & 1u) ? 1u << i : 0u) + ((subm & 2u) ? 1u << j : 0u) + ((subm & 4u) ? 8u : 0u);
                    w[i] = subw[0];
                    w[j] = subw[1];
                    w[imd3[j]] = 0.0f;
                    w[3] = subw[2];
                }
            }
        }
        if (mindist < 0.0f) {
            mindist = 0.0f;
            m = 15;
            w[0] = epa2_det(c, b, d) / vl;
            w[1] = epa2_det(a, c, d) / vl;
            w[2] = epa2_det(b, a, d) / vl;
            w[3] = 1.0f - (w[0] + w[1] + w[2]);
        }
        return mindist;
    }
    return -1.0f;
}

// GJK::Evaluate
template <bool BOX = false>
RLMAX_HD int epa2_gjk(Epa2& g, V3 guess) {
    uint32_t iterations = 0;
    float sqdist = 0.0f, alpha = 0.0f;
    V3 lastw[4];
    uint32_t clastw = 0;
    g.free_[0] = 0;
    g.free_[1] = 1;
    g.free_[2] = 2;
    g.free_[3] = 3;
    g.nfree = 4;
    g.current = 0;
    g.status = 0;
    g.distance = 0.0f;
    g.simplices[0].rank = 0;
    g.ray = guess;
    const float sqrl = len2(g.ray);
    epa2_append<BOX>(g, g.simplices[0], sqrl > 0.0f ? -g.ray : v3(1, 0, 0));
    g.simplices[0].p[0] = 1.0f;
    g.ray = g.sv[g.simplices[0].c[0]].w;
    sqdist = sqrl;
    lastw[0] = lastw[1] = lastw[2] = lastw[3] = g.ray;
    do {
        const int next = 1 - g.current;
        Epa2Simplex& cs = g.simplices[g.current];
        Epa2Simplex& ns = g.simplices[next];
        const float rl = len(g.ray);
        if (rl < EPA2_GJK_MIN_DISTANCE) {
            g.status = 1;
            break;
        }
        epa2_append<BOX>(g, cs, -g.ray);
        const V3 w = g.sv[cs.c[cs.rank - 1]].w;
        bool found = false;
        for (int i = 0; i < 4; ++i)
            if (len2(w - lastw[i]) < EPA2_GJK_DUPLICATED_EPS) {
                found = true;
                break;
            }
        if (found) {
            epa2_remove(g, g.simplices[g.current]);
            break;
        }
        clastw = (clastw + 1) & 3;
        lastw[clastw] = w;
        const float omega = dot(g.ray, w) / rl;
        alpha = fmax_(omega, alpha);
        if (((rl - alpha) - (EPA2_GJK_ACCURACY * rl)) <= 0.0f) {
            epa2_remove(g, g.simplices[g.current]);
            break;
        }
        float weights[4] = {0, 0, 0, 0};
        uint32_t mask = 0;
        const V3 w0 = g.sv[cs.c[0]].w;
        if (cs.rank == 2) sqdist = epa2_proj2(w0, g.sv[cs.c[1]].w, weights, mask);
        else if (cs.rank == 3) sqdist = epa2_proj3(w0, g.sv[cs.c[1]].w, g.sv[cs.c[2]].w, weights, mask);
        else if (cs.rank == 4)
            sqdist = epa2_proj4(w0, g.sv[cs.c[1]].w, g.sv[cs.c[2]].w, g.sv[cs.c[3]].w, weights, mask);
        if (sqdist >= 0.0f) {
            ns.rank = 0;
            g.ray = v3s(0.0f);
            g.current = next;
            for (int i = 0, ni = cs.rank; i < ni; ++i) {
                if (mask & (1u << i)) {
                    ns.c[ns.rank] = cs.c[i];
                    ns.p[ns.rank++] = weights[i];
                    g.ray += g.sv[cs.c[i]].w * weights[i];
                } else {
                    g.free_[g.nfree++] = cs.c[i];
                }
            }
            if (mask == 15) g.status = 1;
        } else {
            epa2_remove(g, g.simplices[g.current]);
            break;
        }
        g.status = (++iterations) < (uint32_t)EPA2_GJK_MAX_ITER ? g.status : 2;
    } while (g.status == 0);
    g.simplex = g.current;
    if (g.status == 0) g.distance = len(g.ray);
    else if (g.status == 1) g.distance = 0.0f;
    return g.status;
}

// GJK::EncloseOrigin without recursion. Bullet recurses once per added vertex, so the simplex rank is the depth:
// each rank below 4 keeps the index of its next candidate direction (rank 1: +-x, +-y, +-z; rank 2: +-(d x axis) for
// each axis, skipped when zero; rank 3: +-n when nonzero), tried in Bullet's order; a rank-4 simplex encloses the
// origin when its determinant is nonzero, else the last vertex is removed and the previous rank tries its next
// candidate. (Bullet's depth guard, depth > 4, cannot fire: the depth never exceeds 4 - rank.)
template <bool BOX = false>
RLMAX_HD bool epa2_enclose(Epa2& g) {
    Epa2Simplex& s = g.simplices[g.simplex];
    const uint32_t base = s.rank;
    int next[4] = {0, 0, 0, 0};
    for (;;) {
        const uint32_t r = s.rank;
        if (r == 4) {
            if (absf(epa2_det(g.sv[s.c[0]].w - g.sv[s.c[3]].w, g.sv[s.c[1]].w - g.sv[s.c[3]].w,
                              g.sv[s.c[2]].w - g.sv[s.c[3]].w)) > 0.0f)
                return true;
        } else {
            V3 dir = v3s(0.0f);
            bool found = false;
            while (!found && next[r] < (r == 3 ? 2 : 6)) {
                const int j = next[r]++;
                V3 axis = v3s(0.0f);
                if (j / 2 == 0) axis.x = 1.0f;
                if (j / 2 == 1) axis.y = 1.0f;
                if (j / 2 == 2) axis.z = 1.0f;
                V3 c = axis;
                if (r == 2) c = cross(g.sv[s.c[1]].w - g.sv[s.c[0]].w, axis);
                if (r == 3) c = cross(g.sv[s.c[1]].w - g.sv[s.c[0]].w, g.sv[s.c[2]].w - g.sv[s.c[0]].w);
                found = r == 1 || len2(c) > 0.0f;
                dir = (j & 1) ? -c : c;
            }
            if (found) {
                epa2_append<BOX>(g, s, dir);
                if (s.rank < 4) next[s.rank] = 0;
                continue;
            }
        }
        // this rank has no candidate left (or the tetrahedron is flat): back to the previous rank
        if (r == base) return false;
        epa2_remove(g, s);
    }
}

struct EpaFace {
    V3 n;
    float d;
    int8_t c[3];  // vertex pool indices, below 4 + EPA_MAX_VERTICES
    int8_t f[3];  // adjacent faces, below EPA_MAX_FACES
    int8_t l[2];  // list links; -1 ends a list
    uint8_t e[3];
    uint8_t pass;
};
static_assert(4 + EPA_MAX_VERTICES <= 127 && EPA_MAX_FACES <= 128, "a face holds its indices in bytes");
struct EpaList {
    int root;
    int count;
};
struct EpaHorizon {
    int cf, ff, nf;
};

enum EpaStatus : int {
    EPA_VALID = 0,
    EPA_TOUCHING = 1,
    EPA_DEGENERATED = 2,
    EPA_NONCONVEX = 3,
    EPA_INVALID_HULL = 4,
    EPA_OUT_OF_FACES = 5,
    EPA_OUT_OF_VERTICES = 6,
    EPA_ACCURACY_REACHED = 7,
    EPA_FALLBACK = 8,
    EPA_FAILED = 9
};

struct Epa {
    EpaFace fc[EPA_MAX_FACES];
    EpaList hull, stock;
    int nextsv;
    int status;
    V3 normal;
    float depth;
    Epa2Simplex result;
};

RLMAX_HD void epa_bind(Epa& e, int fa, uint32_t ea, int fb, uint32_t eb) {
    e.fc[fa].e[ea] = (uint8_t)eb;
    e.fc[fa].f[ea] = fb;
    e.fc[fb].e[eb] = (uint8_t)ea;
    e.fc[fb].f[eb] = fa;
}
RLMAX_HD void epa_append(Epa& e, EpaList& list, int face) {
    e.fc[face].l[0] = -1;
    e.fc[face].l[1] = list.root;
    if (list.root >= 0) e.fc[list.root].l[0] = face;
    list.root = face;
    ++list.count;
}
RLMAX_HD void epa_remove(Epa& e, EpaList& list, int face) {
    if (e.fc[face].l[1] >= 0) e.fc[e.fc[face].l[1]].l[0] = e.fc[face].l[0];
    if (e.fc[face].l[0] >= 0) e.fc[e.fc[face].l[0]].l[1] = e.fc[face].l[1];
    if (face == list.root) list.root = e.fc[face].l[1];
    --list.count;
}
RLMAX_HD void epa_init(Epa& e) {
    e.status = EPA_FAILED;
    e.normal = v3s(0.0f);
    e.depth = 0.0f;
    e.nextsv = 0;
    e.hull = EpaList{-1, 0};
    e.stock = EpaList{-1, 0};
    for (int i = 0; i < EPA_MAX_FACES; ++i) epa_append(e, e.stock, EPA_MAX_FACES - i - 1);
}

RLMAX_HD bool epa_edgedist(const Epa2& g, const EpaFace& face, int a, int b, float& dist) {
    const V3 aw = g.sv[a].w, bw = g.sv[b].w;
    const V3 ba = bw - aw;
    const V3 n_ab = cross(ba, face.n);
    const float a_dot_nab = dot(aw, n_ab);
    if (a_dot_nab < 0.0f) {
        const float ba_l2 = len2(ba);
        const float a_dot_ba = dot(aw, ba);
        const float b_dot_ba = dot(bw, ba);
        if (a_dot_ba > 0.0f) {
            dist = len(aw);
        } else if (b_dot_ba < 0.0f) {
            dist = len(bw);
        } else {
            const float a_dot_b = dot(aw, bw);
            dist = sqrtf(fmax_((len2(aw) * len2(bw) - a_dot_b * a_dot_b) / ba_l2, 0.0f));
        }
        return true;
    }
    return false;
}

RLMAX_HD int epa_newface(Epa& e, const Epa2& g, int a, int b, int c, bool forced) {
    if (e.stock.root >= 0) {
        const int fi = e.stock.root;
        epa_remove(e, e.stock, fi);
        epa_append(e, e.hull, fi);
        EpaFace& face = e.fc[fi];
        face.pass = 0;
        face.c[0] = a;
        face.c[1] = b;
        face.c[2] = c;
        face.n = cross(g.sv[b].w - g.sv[a].w, g.sv[c].w - g.sv[a].w);
        const float l = len(face.n);
        const bool v = l > EPA_ACCURACY;
        if (v) {
            if (!(epa_edgedist(g, face, a, b, face.d) || epa_edgedist(g, face, b, c, face.d) ||
                  epa_edgedist(g, face, c, a, face.d)))
                face.d = dot(g.sv[a].w, face.n) / l;
            face.n = bdiv(face.n, l);
            if (forced || face.d >= -EPA_PLANE_EPS) return fi;
            e.status = EPA_NONCONVEX;
        } else {
            e.status = EPA_DEGENERATED;
        }
        epa_remove(e, e.hull, fi);
        epa_append(e, e.stock, fi);
        return -1;
    }
    e.status = e.stock.root >= 0 ? EPA_OUT_OF_VERTICES : EPA_OUT_OF_FACES;
    return -1;
}

RLMAX_HD int epa_findbest(const Epa& e) {
    int minf = e.hull.root;
    float mind = e.fc[minf].d * e.fc[minf].d;
    for (int f = e.fc[minf].l[1]; f >= 0; f = e.fc[f].l[1]) {
        const float sqd = e.fc[f].d * e.fc[f].d;
        if (sqd < mind) {
            minf = f;
            mind = sqd;
        }
    }
    return minf;
}

// EPA::expand without recursion: the recursion's call stack made explicit, so the frame is static. Each entry is a
// face whose children are being expanded (face | edge << 8, plus 1 << 10 once on its second child); every face on the
// stack carries this pass's mark, so there are at most EPA_MAX_FACES entries. Same order as the recursion: a face's
// mark is tested as it is entered, its second child's link is read only after the first child returns, the second
// child runs only if the first returned true (&&), and the face is retired after both.
RLMAX_HD bool epa_expand(Epa& e, const Epa2& g, uint32_t pass, int w, int f, uint32_t ed, EpaHorizon& hz) {
    const uint32_t i1m3[3] = {1, 2, 0}, i2m3[3] = {2, 0, 1};
    uint16_t stack[EPA_MAX_FACES];
    int sp = 0;
    for (;;) {
        // enter expand(f, ed)
        bool r = false;
        if (e.fc[f].pass != (uint8_t)pass) {
            const uint32_t e1 = i1m3[ed];
            if ((dot(e.fc[f].n, g.sv[w].w) - e.fc[f].d) < -EPA_PLANE_EPS) {
                const int nf = epa_newface(e, g, e.fc[f].c[e1], e.fc[f].c[ed], w, false);
                if (nf >= 0) {
                    epa_bind(e, nf, 0, f, ed);
                    if (hz.cf >= 0) epa_bind(e, hz.cf, 1, nf, 2);
                    else hz.ff = nf;
                    hz.cf = nf;
                    ++hz.nf;
                    r = true;
                }
            } else {
                e.fc[f].pass = (uint8_t)pass;
                stack[sp++] = (uint16_t)((uint32_t)f | ed << 8);
                const int child = e.fc[f].f[e1];
                ed = e.fc[f].e[e1];
                f = child;
                continue;
            }
        }
        // return r to the waiting faces
        for (;;) {
            if (sp == 0) return r;
            const uint32_t top = stack[sp - 1];
            const int pf = (int)(top & 0xffu);
            if (!(top & 0x400u)) {
                if (!r) {  // the first child failed: the face returns false without its second child
                    --sp;
                    continue;
                }
                const uint32_t e2 = i2m3[(top >> 8) & 3u];
                stack[sp - 1] = (uint16_t)(top | 0x400u);
                f = e.fc[pf].f[e2];
                ed = e.fc[pf].e[e2];
                break;
            }
            --sp;
            if (r) {  // both children true: the face leaves the hull
                epa_remove(e, e.hull, pf);
                epa_append(e, e.stock, pf);
            }
        }
    }
}

template <bool BOX = false>
RLMAX_HD int epa_evaluate(Epa& e, Epa2& g, V3 guess) {
    Epa2Simplex& simplex = g.simplices[g.simplex];
    if (simplex.rank > 1 && epa2_enclose<BOX>(g)) {
        while (e.hull.root >= 0) {
            const int f = e.hull.root;
            epa_remove(e, e.hull, f);
            epa_append(e, e.stock, f);
        }
        e.status = EPA_VALID;
        e.nextsv = 0;
        if (epa2_det(g.sv[simplex.c[0]].w - g.sv[simplex.c[3]].w, g.sv[simplex.c[1]].w - g.sv[simplex.c[3]].w,
                     g.sv[simplex.c[2]].w - g.sv[simplex.c[3]].w) < 0.0f) {
            const int tc = simplex.c[0];
            simplex.c[0] = simplex.c[1];
            simplex.c[1] = tc;
            const float tp = simplex.p[0];
            simplex.p[0] = simplex.p[1];
            simplex.p[1] = tp;
        }
        const int tetra[4] = {epa_newface(e, g, simplex.c[0], simplex.c[1], simplex.c[2], true),
                              epa_newface(e, g, simplex.c[1], simplex.c[0], simplex.c[3], true),
                              epa_newface(e, g, simplex.c[2], simplex.c[1], simplex.c[3], true),
                              epa_newface(e, g, simplex.c[0], simplex.c[2], simplex.c[3], true)};
        if (e.hull.count == 4) {
            int best = epa_findbest(e);
            EpaFace outer = e.fc[best];
            uint32_t pass = 0;
            epa_bind(e, tetra[0], 0, tetra[1], 0);
            epa_bind(e, tetra[0], 1, tetra[2], 0);
            epa_bind(e, tetra[0], 2, tetra[3], 0);
            epa_bind(e, tetra[1], 1, tetra[3], 2);
            epa_bind(e, tetra[1], 2, tetra[2], 1);
            epa_bind(e, tetra[2], 2, tetra[3], 1);
            e.status = EPA_VALID;
            for (int it = 0; it < EPA_MAX_ITERATIONS; ++it) {
                if (e.nextsv < EPA_MAX_VERTICES) {
                    EpaHorizon hz{-1, -1, 0};
                    const int w = 4 + e.nextsv++;
                    bool valid = true;
                    e.fc[best].pass = (uint8_t)(++pass);
                    epa2_getsupport<BOX>(g, e.fc[best].n, g.sv[w]);
                    const float wdist = dot(e.fc[best].n, g.sv[w].w) - e.fc[best].d;
                    if (wdist > EPA_ACCURACY) {
                        for (uint32_t j = 0; j < 3 && valid; ++j)
                            valid &= epa_expand(e, g, pass, w, e.fc[best].f[j], e.fc[best].e[j], hz);
                        if (valid && hz.nf >= 3) {
                            epa_bind(e, hz.cf, 1, hz.ff, 2);
                            epa_remove(e, e.hull, best);
                            epa_append(e, e.stock, best);
                            best = epa_findbest(e);
                            outer = e.fc[best];
                        } else {
                            e.status = EPA_INVALID_HULL;
                            break;
                        }
                    } else {
                        e.status = EPA_ACCURACY_REACHED;
                        break;
                    }
                } else {
                    e.status = EPA_OUT_OF_VERTICES;
                    break;
                }
            }
            const V3 projection = outer.n * outer.d;
            e.normal = outer.n;
            e.depth = outer.d;
            e.result.rank = 3;
            e.result.c[0] = outer.c[0];
            e.result.c[1] = outer.c[1];
            e.result.c[2] = outer.c[2];
            e.result.p[0] = len(cross(g.sv[outer.c[1]].w - projection, g.sv[outer.c[2]].w - projection));
            e.result.p[1] = len(cross(g.sv[outer.c[2]].w - projection, g.sv[outer.c[0]].w - projection));
            e.result.p[2] = len(cross(g.sv[outer.c[0]].w - projection, g.sv[outer.c[1]].w - projection));
            const float sum = e.result.p[0] + e.result.p[1] + e.result.p[2];
            e.result.p[0] /= sum;
            e.result.p[1] /= sum;
            e.result.p[2] /= sum;
            return e.status;
        }
    }
    e.status = EPA_FALLBACK;
    e.normal = -guess;
    const float nl = len(e.normal);
    e.normal = nl > 0.0f ? bdiv(e.normal, nl) : v3(1, 0, 0);
    e.depth = 0.0f;
    e.result.rank = 1;
    e.result.c[0] = simplex.c[0];
    e.result.p[0] = 1.0f;
    return e.status;
}

struct PenResult {
    bool pen;   // Penetration succeeded
    bool dist;  // else Distance succeeded
    V3 wa, wb;  // witnesses, in the frame of the transforms passed
    V3 v;       // results.normal
};

// btGjkEpaPenetrationDepthSolver::calcPenDepth for the box (basis ra, origin ta) against the triangle (basis identity,
// origin tb): Penetration with margins, else Distance without; the guess is tb - ta
// EPA's working storage (the GJK simplex pool and the polytope, ~9 KB), per arena: global memory on the GPU, touched
// only when the penetration solver runs, so it stays out of the hot tick's frame.
struct EpaScratch {
    Epa2 g;
    Epa e;
};

// The penetration path is uncommon; keep EPA out of the tick's instruction stream.
RLMAX_COLD PenResult epa_box_tri(V3 h, float margin, const M3& ra, V3 ta, const V3 tv[3], V3 tb, EpaScratch& sc) {
    PenResult r{false, false, v3s(0.0f), v3s(0.0f), v3s(0.0f)};
    const V3 guess = tb - ta;
    Epa2& g = sc.g;
    g.shape.h = h;
    g.shape.margin = margin;
    g.shape.to1 = ra;  // B^T A with B the identity
    g.shape.to0_b = M3{v3(ra.c0.x, ra.c1.x, ra.c2.x), v3(ra.c0.y, ra.c1.y, ra.c2.y), v3(ra.c0.z, ra.c1.z, ra.c2.z)};
    g.shape.to0_o = mul_t(ra, tb - ta);
    for (int i = 0; i < 3; i++) g.shape.tv[i] = tv[i];
    g.shape.with_margin = true;
    if (epa2_gjk(g, -guess) == 1) {
        Epa& e = sc.e;
        epa_init(e);
        if (epa_evaluate(e, g, -guess) != EPA_FAILED) {
            V3 w0 = v3s(0.0f);
            for (int i = 0; i < e.result.rank; ++i) w0 += epa2_support0(g.shape, g.sv[e.result.c[i]].d) * e.result.p[i];
            r.pen = true;
            r.wa = ra * w0 + ta;
            r.wb = ra * (w0 - e.normal * e.depth) + ta;
            r.v = -e.normal;
            return r;
        }
    }
    // Distance: GJK without margins (a fresh GJK on the same shapes: epa2_gjk re-initialises everything but the shape)
    Epa2& d = g;
    d.shape.with_margin = false;
    if (epa2_gjk(d, guess) == 0) {
        Epa2Simplex& s = d.simplices[d.simplex];
        V3 w0 = v3s(0.0f), w1 = v3s(0.0f);
        for (int i = 0; i < s.rank; ++i) {
            const float p = s.p[i];
            w0 += epa2_support0(d.shape, d.sv[s.c[i]].d) * p;
            w1 += epa2_support1(d.shape, -d.sv[s.c[i]].d) * p;
        }
        r.dist = true;
        r.wa = ra * w0 + ta;
        r.wb = ra * w1 + ta;
        const V3 n = w0 - w1;
        const float dl = len(n);
        r.v = bdiv(n, dl > EPA2_GJK_MIN_DISTANCE ? dl : 1.0f);
    }
    return r;
}

// The same penetration solver for two boxes. Only the second support map differs from the box-triangle cast;
// the template keeps that choice out of the triangle path's runtime loop.
RLMAX_COLD PenResult epa_box_box(V3 ha, float ma, const M3& ra, V3 ta, V3 hb, float mb, const M3& rb, V3 tb,
                                 EpaScratch& sc) {
    PenResult r{false, false, v3s(0.0f), v3s(0.0f), v3s(0.0f)};
    Epa2& g = sc.g;
    g.shape.h = ha; g.shape.margin = ma; g.shape.h2 = hb; g.shape.margin2 = mb;
    g.shape.to1 = M3{mul_t(rb, ra.c0), mul_t(rb, ra.c1), mul_t(rb, ra.c2)};
    g.shape.to0_b = M3{mul_t(ra, rb.c0), mul_t(ra, rb.c1), mul_t(ra, rb.c2)};
    g.shape.to0_o = mul_t(ra, tb - ta);
    const V3 guess = tb - ta;
    g.shape.with_margin = true;
    if (epa2_gjk<true>(g, -guess) == 1) {
        Epa& e = sc.e;
        epa_init(e);
        if (epa_evaluate<true>(e, g, -guess) != EPA_FAILED) {
            V3 w = v3s(0.0f);
            for (int i = 0; i < e.result.rank; i++) w += epa2_support0(g.shape, g.sv[e.result.c[i]].d) * e.result.p[i];
            r.pen = true;
            r.wa = ra * w + ta;
            r.wb = ra * (w - e.normal * e.depth) + ta;
            r.v = -e.normal;
            return r;
        }
    }
    g.shape.with_margin = false;
    if (epa2_gjk<true>(g, guess) == 0) {
        const Epa2Simplex& s = g.simplices[g.simplex];
        V3 wa = v3s(0.0f), wb = v3s(0.0f);
        for (int i = 0; i < s.rank; i++) {
            const V3 d = g.sv[s.c[i]].d;
            wa += epa2_support0(g.shape, d) * s.p[i];
            wb += epa2_support1<true>(g.shape, -d) * s.p[i];
        }
        r.dist = true;
        r.wa = ra * wa + ta; r.wb = ra * wb + ta;
        const V3 n = wa - wb;
        const float l = len(n);
        r.v = bdiv(n, l > EPA2_GJK_MIN_DISTANCE ? l : 1.0f);
    }
    return r;
}

struct GjkResult {
    bool valid;       // a distance result (shapes without margins separated, or touching)
    bool contact;     // ... within the maximum distance: a contact
    bool from_epa;    // the result came from the penetration solver
    bool wants_epa;   // Bullet would run the penetration solver (no valid result, or the degenerate-case catch)
    V3 pb;            // witness on the triangle (world)
    float d;          // distance minus the margins
};

// Box A (implicit half extents h, margin ma, basis ra, origin oa) against triangle B (vertices tv in B's frame, basis
// identity, origin ob, margin 0); max_dist2 = (ma + breaking threshold)^2.
// Plane-rejected candidates do not call the box-triangle distance solver.
RLMAX_COLD GjkResult gjk_box_tri(V3 h, float ma, const M3& ra, V3 oa, const V3 tv[3], V3 ob, float max_dist2,
                                EpaScratch& sc) {
    GjkResult res{false, false, false, false, v3s(0.0f), 0.0f};
    const float mb = 0.0f;
    const V3 off = (oa + ob) * 0.5f;
    const V3 la = oa - off, lb = ob - off;
    V3 axis = v3(0, 1, 0);
    float sq = GJK_LARGE;
    const float margin = ma + mb;
    GjkSimplex s;
    gjk_reset(s);
    bool check_simplex = false;
    int degenerate = 0;
    int iter = 0;
    for (int guard = 0; guard < GJK_MAX_ITER + 2; guard++) {
        V3 dir_a = mul_t(ra, -axis);
        V3 dir_b = axis;
        V3 pw = ra * gjk_box_support(h, dir_a) + la;
        V3 qw = gjk_tri_support(tv, dir_b) + lb;
        V3 w = pw - qw;
        float delta = dot(axis, w);
        if (delta > 0.0f && delta * delta > sq * max_dist2) {
            degenerate = 10;
            check_simplex = true;
            break;
        }
        if (gjk_in_simplex(s, w)) {
            degenerate = 1;
            check_simplex = true;
            break;
        }
        float f0 = sq - delta, f1 = sq * GJK_REL_ERROR2;
        if (f0 <= f1) {
            degenerate = f0 <= 0.0f ? 2 : 11;
            check_simplex = true;
            break;
        }
        gjk_add(s, w, pw, qw);
        if (!gjk_update(s)) {
            degenerate = 3;
            check_simplex = true;
            break;
        }
        V3 nax = s.cv;
        if (len2(nax) < GJK_REL_ERROR2) {
            axis = nax;
            degenerate = 6;
            check_simplex = true;
            break;
        }
        float prev = sq;
        sq = len2(nax);
        if (prev - sq <= FLT_EPS * prev) {
            check_simplex = true;
            degenerate = 12;
            break;
        }
        axis = nax;
        if (iter++ > GJK_MAX_ITER) break;
        if (s.n == 4) {
            degenerate = 13;
            break;
        }
    }
    float distance = 0.0f;
    V3 pa = v3s(0.0f), pb = v3s(0.0f);
    bool valid = false;
    if (check_simplex) {
        gjk_update(s);
        pa = s.cp1;
        pb = s.cp2;
        float l2 = len2(axis);
        if (l2 > FLT_EPS * FLT_EPS) {
            float rlen = 1.0f / sqrtf(l2);
            float sd = sqrtf(sq);
            pa -= axis * (ma / sd);
            pb += axis * (mb / sd);
            distance = (1.0f / rlen) - margin;
            valid = true;
        }
    }
    res.wants_epa = !valid || (degenerate != 0 && (distance + margin) < 0.01f);
    if (res.wants_epa && quirk::EPA_BOX_TRI) {
        // the penetration solver, in the same shifted frame; only a deeper result replaces a valid one
        PenResult pr = epa_box_tri(h, ma, ra, la, tv, lb, sc);
        if (pr.pen) {
            V3 tn = pr.wb - pr.wa;
            float l2n = len2(tn);
            if (l2n <= FLT_EPS * FLT_EPS) l2n = len2(pr.v);
            if (l2n > FLT_EPS * FLT_EPS) {
                const float d2 = -len(pr.wa - pr.wb);
                if (!valid || d2 < distance) {
                    distance = d2;
                    pa = pr.wa;
                    pb = pr.wb;
                    valid = true;
                    res.from_epa = true;
                }
            }
        } else if (pr.dist && len2(pr.v) > 0.0f) {
            const float d2 = len(pr.wa - pr.wb) - margin;
            if (!valid || d2 < distance) {
                distance = d2;
                pa = pr.wa - pr.v * ma;
                pb = pr.wb + pr.v * mb;
                valid = true;
                res.from_epa = true;
            }
        }
    }
    (void)pa;
    res.valid = valid;
    if (valid && (distance < 0.0f || distance * distance < max_dist2)) {
        res.contact = true;
        res.pb = pb + off;
        res.d = distance;
    }
    return res;
}

// btGjkPairDetector of the car's box (A) against the ball (B), as the game runs it. A sphere's
// support without margin is its centre, its margin its radius. `ok` is false when the penetration solver would run (the centre inside
// the box): the caller keeps the analytic contact for that.
struct GjkSphereResult { bool ok, contact; V3 normal, on_ball; float d; };
// Only overlapping ball-car boxes reach this distance solver.
RLMAX_COLD GjkSphereResult gjk_box_sphere(V3 h, float ma, const M3& ra, V3 oa, V3 ob, float mb, float max_dist2) {
    GjkSphereResult res{false, false, v3s(0.0f), v3s(0.0f), 0.0f};
    const V3 off = (oa + ob) * 0.5f;
    const V3 la = oa - off, lb = ob - off;
    V3 axis = v3(0, 1, 0);
    float sq = GJK_LARGE;
    const float margin = ma + mb;
    GjkSimplex s;
    gjk_reset(s);
    bool check_simplex = false;
    int degenerate = 0;
    int iter = 0;
    for (int guard = 0; guard < GJK_MAX_ITER + 2; guard++) {
        V3 pw = ra * gjk_box_support(h, mul_t(ra, -axis)) + la;
        V3 w = pw - lb;
        float delta = dot(axis, w);
        if (delta > 0.0f && delta * delta > sq * max_dist2) { degenerate = 10; check_simplex = true; break; }
        if (gjk_in_simplex(s, w)) { degenerate = 1; check_simplex = true; break; }
        float f0 = sq - delta, f1 = sq * GJK_REL_ERROR2;
        if (f0 <= f1) { degenerate = f0 <= 0.0f ? 2 : 11; check_simplex = true; break; }
        gjk_add(s, w, pw, lb);
        if (!gjk_update(s)) { degenerate = 3; check_simplex = true; break; }
        V3 nax = s.cv;
        if (len2(nax) < GJK_REL_ERROR2) { axis = nax; degenerate = 6; check_simplex = true; break; }
        float prev = sq;
        sq = len2(nax);
        if (prev - sq <= FLT_EPS * prev) { check_simplex = true; degenerate = 12; break; }
        axis = nax;
        if (iter++ > GJK_MAX_ITER) break;
        if (s.n == 4) { degenerate = 13; break; }
    }
    if (!check_simplex) return res;
    gjk_update(s);
    V3 pb = s.cp2;
    const float l2 = len2(axis);
    if (!(l2 > FLT_EPS * FLT_EPS)) return res;
    const float sd = sqrtf(sq);
    if (!(sd > 0.0f)) return res;
    const float rlen = 1.0f / sqrtf(l2);
    pb += axis * (mb / sd);
    const float distance = (1.0f / rlen) - margin;
    if (degenerate != 0 && (distance + margin) <= 0.01f) return res;
    res.ok = true;
    res.normal = axis * rlen;
    res.on_ball = pb + off;
    res.d = distance;
    res.contact = distance < 0.0f || distance * distance < max_dist2;
    return res;
}

}  // namespace rlmax
