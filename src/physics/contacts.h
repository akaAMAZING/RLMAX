// RLMAX narrowphase: fixed-cost contact generation (DESIGN §4.2-4.3). Semantics follow the game's Bullet as modelled in
// the earlier parity work; structure is RLMAX's own: every routine is branch-light and loop-free or fixed-trip.
#pragma once

#include "arena_data.h"
#include "consts.h"
#include "gjk.h"
#include "state.h"

namespace rlmax {

// A contact between body A (dynamic) and B. `pb` is the point on B, `n` the normal on B (pointing towards A), `d` the
// signed distance (negative = penetration); the point on A is pb + n * d.
struct Contact {
    V3 pb;
    V3 n;
    float d;
    bool ok;
    uint32_t group;  // the manifold a point belongs to: mesh index, or 0x100 + plane index
};

RLMAX_HD Contact no_contact() { return Contact{v3s(0), v3(0, 0, 1), 0.0f, false, 0u}; }

// ---- sphere vs triangle (bullet/collision/shapes/triangle_shape.rs intersect_sphere_front_precomputed) ----------

RLMAX_HD float segment_sqr_distance(V3 from, V3 to, V3 p, V3* nearest) {
    V3 diff = p - from;
    V3 v = to - from;
    float t = dot(v, diff);
    if (t > 0.0f) {
        float dvv = dot(v, v);
        if (t < dvv) {
            t /= dvv;
            diff -= v * t;
        } else {
            t = 1.0f;
            diff -= v;
        }
    } else {
        t = 0.0f;
    }
    *nearest = from + v * t;
    return dot(diff, diff);
}

// A goal triangle is tested in its body's frame, as the game's: the game's local vertices, the centre less the origin, the
// normal from the local vertices; the point on the triangle comes back through the origin.
RLMAX_HD Contact sphere_tri(const Tri& t, V3 c, float radius, float thresh, const float* goal_ly) {
    Contact out = no_contact();
    V3 p0 = t.p0, p1 = t.p1, p2 = t.p2, tn = t.n;
    const V3 org = tri_origin(t);
    if (t.flags & TF_ORIGIN_GOAL) {
        V3 lv[3];
        goal_tri_local(t, goal_ly, lv);
        p0 = lv[0];
        p1 = lv[1];
        p2 = lv[2];
        c = v3(c.x, c.y - org.y, c.z);
        tn = cross(p1 - p0, p2 - p0);
        tn = tn * (1.0f / sqrtf(dot(tn, tn)));
    }
    V3 to_center = c - p0;
    float dist_plane = dot(to_center, tn);
    float rwt = radius + thresh;
    float rwt2 = rwt * rwt;
    if (dist_plane < 0.0f || dist_plane >= rwt) return out;
    // face_contains: edges e0 = p1-p0, e1 = p2-p1, e2 = p0-p2 against (c - p_i)
    V3 o0 = to_center, o1 = c - p1, o2 = c - p2;
    bool inside = dot(cross(p1 - p0, o0), tn) >= 0.0f && dot(cross(p2 - p1, o1), tn) >= 0.0f &&
                  dot(cross(p0 - p2, o2), tn) >= 0.0f;
    V3 cp;
    if (inside) {
        cp = c - tn * dist_plane;
    } else {
        float best = rwt2;
        V3 e;
        cp = v3s(0.0f);
        float d0 = segment_sqr_distance(p0, p1, c, &e);
        if (d0 < best) { best = d0; cp = e; }
        float d1 = segment_sqr_distance(p1, p2, c, &e);
        if (d1 < best) { best = d1; cp = e; }
        float d2 = segment_sqr_distance(p2, p0, c, &e);
        if (d2 < best) { best = d2; cp = e; }
        if (!(best < rwt2)) return out;
    }
    V3 to_c = c - cp;
    float d2 = len2(to_c);
    if (d2 >= rwt2) return out;
    if (d2 > FLT_EPS) {
        float dist = sqrtf(d2);
        out.n = to_c * (1.0f / dist);
        out.d = -(radius - dist);
    } else {
        out.n = tn;
        out.d = -radius;
    }
    out.pb = v3(cp.x, cp.y + org.y, cp.z);
    out.ok = true;
    return out;
}

// ---- internal-edge adjustment ------------------------------
// Triangles are stored in world space with the meshes' identity rotation, so "local" equals world here.

constexpr float EDGE_T_MIN = 9.99999974737875e-06f;  // bits 0x3727c5ac
constexpr float EDGE_T_MAX = 0.999989986419678f;     // bits 0x3f7fff58
constexpr float MAX_EDGE_ANGLE = 2.0f * PI;
constexpr float EDGE_DISTANCE_THRESHOLD = 0.1f;

RLMAX_HD V3 nearest_on_segment(V3 p, V3 l0, V3 l1) {
    V3 ld = l1 - l0;
    if (len2(ld) < FLT_EPS * FLT_EPS) return l0;
    float t = dot(p - l0, ld) / dot(ld, ld);
    return l0 + ld * clampf(t, EDGE_T_MIN, EDGE_T_MAX);
}

RLMAX_HD float edge_get_angle(V3 edge_a, V3 normal_a, V3 normal_b) {
    return game_atan2f(dot(normal_b, edge_a), dot(normal_b, normal_a));
}

// btQuaternion(axis, angle): the axis is divided by its length inside the sine, not normalised first.
RLMAX_HD Q bullet_axis_angle(V3 axis, float angle) {
    const float s = ucrt_sinf(angle * 0.5f) / len(axis);
    return Q{axis.x * s, axis.y * s, axis.z * s, ucrt_cosf(angle * 0.5f)};
}

// Returns true and the clamped normal when the contact normal lies outside the edge's allowed angle.
RLMAX_HD bool clamp_normal(V3 edge, V3 tri_normal_org, V3 n_local, float corrected_angle, V3* out) {
    V3 edge_cross = normalize(cross(edge, tri_normal_org));
    float cur = edge_get_angle(edge_cross, tri_normal_org, n_local);
    if ((corrected_angle < 0.0f && cur < corrected_angle) || (corrected_angle >= 0.0f && cur > corrected_angle)) {
        *out = bullet_set_rotation(bullet_axis_angle(edge, corrected_angle - cur)) * n_local;  // btMatrix3x3 * v
        return true;
    }
    return false;
}

// One edge's adjustment. `edge_idx` 0: v0v1, 1: v1v2, 2: v2v0.
RLMAX_HD void adjust_edge(const Tri& t, int edge_idx, float ang, uint32_t convex_bit, uint32_t swap_bit, V3 pa, float d,
                          V3 n_orig, V3* n_world, V3* pb, bool* concave_hit) {
    if (ang == 0.0f) {
        *concave_hit = true;
        return;
    }
    V3 e = edge_idx == 0 ? (t.p1 - t.p0) : (edge_idx == 1 ? (t.p2 - t.p1) : (t.p0 - t.p2));
    V3 edge = -e;
    float swap_factor = (t.flags & convex_bit) ? 1.0f : -1.0f;
    V3 n_a = t.n * swap_factor;
    // quatRotate: q * v * q^-1 as btQuaternion multiplies it
    const Q rq = bullet_axis_angle(edge, ang);
    const float rx = rq.w * t.n.x + rq.y * t.n.z - rq.z * t.n.y, ry = rq.w * t.n.y + rq.z * t.n.x - rq.x * t.n.z;
    const float rz = rq.w * t.n.z + rq.x * t.n.y - rq.y * t.n.x, rw = -rq.x * t.n.x - rq.y * t.n.y - rq.z * t.n.z;
    V3 computed_nb = v3(rw * -rq.x + rx * rq.w + ry * -rq.z - rz * -rq.y, rw * -rq.y + ry * rq.w + rz * -rq.x - rx * -rq.z,
                        rw * -rq.z + rz * rq.w + rx * -rq.y - ry * -rq.x);
    if (t.flags & swap_bit) computed_nb = -computed_nb;
    V3 n_b = computed_nb * swap_factor;
    // the earlier model tests back-facing against the normal from before any edge adjustment, then clamps the current one.
    if (dot(n_orig, n_a) < 0.0f && dot(n_orig, n_b) < 0.0f) {
        *concave_hit = true;
        return;
    }
    V3 clamped;
    if (clamp_normal(edge, t.n * swap_factor, edge_idx == 0 ? n_orig : *n_world, ang, &clamped) && dot(clamped, t.n) > 0.0f) {
        *n_world = clamped;
        *pb = pa - clamped * d;
    }
}

// A goal triangle is read in its body's frame (the game's local vertices, the normal from them, the contact point less the
// origin); the adjusted point on B comes back in the world.
// Only retained triangle contacts need edge correction.
RLMAX_COLD void adjust_internal_edge(const Tri& world_tri, Contact* c, const float* goal_ly) {
    Tri t = world_tri;
    V3 org = v3s(0.0f);
    if (t.flags & TF_ORIGIN_GOAL) {
        V3 lv[3];
        goal_tri_local(t, goal_ly, lv);
        t.p0 = lv[0];
        t.p1 = lv[1];
        t.p2 = lv[2];
        t.n = normalize(cross(lv[1] - lv[0], lv[2] - lv[0]));
        org = tri_origin(t);
    }
    V3 pa = c->pb + c->n * c->d;  // point on A, unchanged by the adjustment
    V3 contact = v3(c->pb.x, c->pb.y - org.y, c->pb.z);
    float dist[3] = {FLT_BIG, FLT_BIG, FLT_BIG};
    bool in_range[3] = {absf(t.ang[0]) < MAX_EDGE_ANGLE, absf(t.ang[1]) < MAX_EDGE_ANGLE, absf(t.ang[2]) < MAX_EDGE_ANGLE};
    int best = -1;
    float dist_best = FLT_BIG;
    if (in_range[0]) {
        dist[0] = len(contact - nearest_on_segment(contact, t.p0, t.p1));
        if (dist[0] < dist_best) { best = 0; dist_best = dist[0]; }
    }
    if (in_range[1]) {
        dist[1] = len(contact - nearest_on_segment(contact, t.p1, t.p2));
        if (dist[1] < dist_best || (dist[1] == dist_best && t.ang[1] != 0.0f && !(t.flags & TF_V1V2_CONVEX))) {
            best = 1;
            dist_best = dist[1];
        }
    }
    if (in_range[2]) {
        dist[2] = len(contact - nearest_on_segment(contact, t.p2, t.p0));
        if (dist[2] < dist_best || (dist[2] == dist_best && t.ang[2] != 0.0f)) best = 2;
    }
    // the normal before any edge adjustment, normalised once more (the game reads this one for edge 0, the back-facing
    // test and the concave fallback; the later edges clamp the running normal)
    const float n_l2 = len2(c->n);
    V3 n0 = n_l2 > 1.0e-12f ? c->n * (1.0f / sqrtf(n_l2)) : t.n;
    bool processed = false, concave = false;
    const uint32_t cbits[3] = {TF_V0V1_CONVEX, TF_V1V2_CONVEX, TF_V2V0_CONVEX};
    const uint32_t sbits[3] = {TF_V0V1_SWAP, TF_V1V2_SWAP, TF_V2V0_SWAP};
    for (int k = 0; k < 3; k++) {
        if (in_range[k] && dist[k] < EDGE_DISTANCE_THRESHOLD &&
            (best == k || (dist[k] < FLT_EPS && t.ang[k] != 0.0f))) {
            processed = true;
            adjust_edge(t, k, t.ang[k], cbits[k], sbits[k], pa, c->d, n0, &c->n, &c->pb, &concave);
        }
    }
    if (!processed || !concave) return;
    if (dot(t.n, n0) < 0.0f) return;
    c->n = t.n;
    c->pb = pa - t.n * c->d;
}

// ---- oriented boxes --------------------------------------------------------------------------------------------

struct Obb {
    V3 c;       // centre, world
    M3 r;       // axes (columns)
    V3 h;       // implicit half extents (without margin)
    float margin;
};

RLMAX_HD V3 obb_full_half(const Obb& b) { return b.h + v3s(b.margin); }

// A box's world bounding box (btBoxShape::getAabb: btTransformAabb of the half extents plus margin).
RLMAX_HD V3 obb_extent(const Obb& b) {
    const V3 fh = obb_full_half(b);
    return v3(absf(b.r.c0.x) * fh.x + absf(b.r.c1.x) * fh.y + absf(b.r.c2.x) * fh.z,
              absf(b.r.c0.y) * fh.x + absf(b.r.c1.y) * fh.y + absf(b.r.c2.y) * fh.z,
              absf(b.r.c0.z) * fh.x + absf(b.r.c1.z) * fh.y + absf(b.r.c2.z) * fh.z);
}
RLMAX_HD void obb_aabb(const Obb& b, V3* mn, V3* mx) {
    const V3 e = obb_extent(b);
    *mn = b.c - e;
    *mx = b.c + e;
}

// Bullet's TestAabbAgainstAabb2 (bounds inclusive).
RLMAX_HD bool aabb_overlap(V3 amn, V3 amx, V3 bmn, V3 bmx) {
    return !(amn.x > bmx.x || amx.x < bmn.x || amn.y > bmx.y || amx.y < bmn.y || amn.z > bmx.z || amx.z < bmn.z);
}

// Bullet's TestTriangleAgainstAabb2: the triangle's vertex bounds against a box (bounds inclusive).
// Axis order as the game's.
RLMAX_HD bool tri_aabb_overlap(V3 p0, V3 p1, V3 p2, V3 mn, V3 mx) {
    if (fmin_(p0.z, fmin_(p1.z, p2.z)) > mx.z || fmax_(p0.z, fmax_(p1.z, p2.z)) < mn.z) return false;
    if (fmin_(p0.x, fmin_(p1.x, p2.x)) > mx.x || fmax_(p0.x, fmax_(p1.x, p2.x)) < mn.x) return false;
    return !(fmin_(p0.y, fmin_(p1.y, p2.y)) > mx.y || fmax_(p0.y, fmax_(p1.y, p2.y)) < mn.y);
}

// The 4 corners nearest the plane, each in its own fixed slot with `ok` set when within `thresh` (the earlier model adds one support
// vertex per tick to a persistent manifold, which over consecutive ticks holds the same corners; RLMAX generates them in
// one tick). Slot 0 is the support corner, slots 1-3 its neighbours along each box axis.
RLMAX_HD void obb_plane4(const Obb& b, V3 pn, float pd, float thresh, Contact out[4]) {
    V3 fh = obb_full_half(b);
    V3 ln = mul_t(b.r, pn);  // plane normal in box frame
    V3 s = v3(ln.x > 0.0f ? -fh.x : fh.x, ln.y > 0.0f ? -fh.y : fh.y, ln.z > 0.0f ? -fh.z : fh.z);
    V3 corners[4] = {s, v3(-s.x, s.y, s.z), v3(s.x, -s.y, s.z), v3(s.x, s.y, -s.z)};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        V3 w = b.c + b.r * corners[i];
        float dist = dot(pn, w) - pd;
        out[i].pb = w - pn * dist;
        out[i].n = pn;
        out[i].d = dist;
        out[i].ok = dist < thresh;
    }
}

// True when obb_plane4 can mark a corner ok. Each corner is c + r * (+-fh), so its distance to the plane is at least the
// centre's distance minus the box's support radius along the normal, sum |ln_k| fh_k; slot 0 is the nearest and the other
// three are no nearer. A corner is ok only if its distance is < thresh, so when the centre minus the radius is past
// thresh + 0.01 BT (far above float error at arena scale) no corner is ok and obb_plane4 need not run.
RLMAX_HD bool obb_plane_reach(const Obb& b, V3 pn, float pd, float thresh) {
    const V3 fh = obb_full_half(b);
    const V3 ln = mul_t(b.r, pn);
    const float support = dot(vabs(ln), fh);
    return dot(pn, b.c) - pd - support <= thresh + 0.01f;
}

// ---- the game's static planes, as the car's box sees them -------------------------------------------------------------
// The game places each plane through a rotator-built transform whose residuals survive. 0 floor, 1 ceiling,
// 2 +x wall, 3 -x wall: the order of arena_plane; 4, 5 the goal floor patches (z = 0 at y = -+111.68, the -y one first).
// Rows r0..r2 of the plane's rotation, origin o, half sizes of the finite plane in its own x and y (BT).
struct PlaneXf {
    V3 r0, r1, r2, o;
    float hx, hy;
};

RLMAX_HD PlaneXf plane_xf(int p) {
    const float X = c::ARENA_X_UU * c::UU_TO_BT, Z = c::ARENA_Z_UU * c::UU_TO_BT;
    PlaneXf x{v3(1, 0, 0), v3(0, 1, 0), v3(0, 0, 1), v3s(0.0f), 4358.144f / 50.0f, 6225.92f / 50.0f};
    if (p >= 4) {
        x.o = v3(0.0f, p == 4 ? -111.68f : 111.68f, 0.0f);
        x.hx = 17.694719314575195f;
        x.hy = 8.847359657287598f;
    } else if (p == 1) {
        x.o = v3(0.0f, 0.0f, Z);
        x.r0 = v3(-1.0f, -0x1.777a5cp-24f, -0x1.135bd8p-47f);
        x.r1 = v3(-0x1.777a5cp-24f, 1.0f, 0x1.777a5cp-24f);
        x.r2 = v3(-0x1p-71f, 0x1.777a5cp-24f, -1.0f);
    } else if (p == 2) {
        x.o = v3(-X, 0.0f, 992.0f * c::UU_TO_BT);
        x.hx = 1245.184f / 50.0f;
        x.r0 = v3(0.0f, 0.0f, 1.0f);
        x.r1 = v3(0.0f, 1.0f, 0.0f);
        x.r2 = v3(-1.0f, 0.0f, 0.0f);
    } else if (p == 3) {
        x.o = v3(X, 0.0f, 992.0f * c::UU_TO_BT);
        x.hx = 1245.184f / 50.0f;
        x.r0 = v3(0x1.8p-23f, -0x1.8p-70f, -0x1.fffffep-1f);
        x.r1 = v3(0x1.135bd8p-47f, 1.0f, 0.0f);
        x.r2 = v3(0x1.fffffep-1f, -0x1.135bd6p-47f, 0x1.8p-23f);
    }
    return x;
}

// a point stored as an offset from the plane's origin, back in the world
RLMAX_HD V3 plane_to_world(const PlaneXf& x, V3 off) {
    const V3 l = v3(dot(v3(x.r0.x, x.r1.x, x.r2.x), off), dot(v3(x.r0.y, x.r1.y, x.r2.y), off),
                    dot(v3(x.r0.z, x.r1.z, x.r2.z), off));
    return v3(dot(l, x.r0) + x.o.x, dot(l, x.r1) + x.o.y, dot(l, x.r2) + x.o.z);
}

enum PlaneHit { PLANE_FAR, PLANE_OUTSIDE, PLANE_POINT };

// The box's one new point against a plane this tick: its support vertex through the plane's own frame (the float order of each step is the game's, the zero terms of its dots are dropped:
// the normal is the rotation's third column, the projection ignores the in-plane z). PLANE_FAR: the vertex is not within
// `thr` (the plane's manifold is dropped); PLANE_OUTSIDE: it is, but off the finite plane (nothing is added);
// PLANE_POINT: `*out` is the point, la in the box's frame, lb as an offset from the plane origin.
RLMAX_HD PlaneHit plane_point(const PlaneXf& x, const Obb& b, V3 pos, float thr, MPoint* out) {
    const V3 n = v3(x.r0.z, x.r1.z, x.r2.z);
    const V3 fh = obb_full_half(b);
    const V3 ln = mul_t(b.r, n);  // the plane's normal in the box frame
    const V3 v = v3(ln.x <= 0.0f ? fh.x : -fh.x, ln.y <= 0.0f ? fh.y : -fh.y, ln.z <= 0.0f ? fh.z : -fh.z);
    // the vertex in the plane's frame, per axis: the plane's column against the box axes, summed with the origin offset
    const V3 no = v3(-x.o.x, -x.o.y, -x.o.z);
    const float z = ((ln.x * v.x + ln.y * v.y) + ln.z * v.z) + (dot(n, b.c) + dot(n, no));
    if (!(z < thr)) return PLANE_FAR;
    auto in_plane = [&](V3 col) {
        return ((dot(col, b.r.c0) * v.x + dot(col, b.r.c1) * v.y) + dot(col, b.r.c2) * v.z) + (dot(col, b.c) + dot(col, no));
    };
    const float px = in_plane(v3(x.r0.x, x.r1.x, x.r2.x)), py = in_plane(v3(x.r0.y, x.r1.y, x.r2.y));
    // the finite plane
    const float lx = thr + x.hx, ly = thr + x.hy;
    if (px < -lx || py < -ly || lx < px || ly < py) return PLANE_OUTSIDE;
    const V3 on_b = v3((x.r0.x * px + x.r0.y * py) + x.o.x, (x.r1.x * px + x.r1.y * py) + x.o.y,
                       (x.r2.x * px + x.r2.y * py) + x.o.z);
    const V3 rel = (on_b + n * z) - pos;  // the point on the box
    const V3 la = mul_t(b.r, rel);
    const V3 lb = on_b - x.o;
    *out = MPoint{la, lb, n, dot((pos + b.r * la) - plane_to_world(x, lb), n)};
    return PLANE_POINT;
}

// The ball (centre c, basis r) against plane p, as the game's convex-plane algorithm makes it.
// The sphere's support point is taken in the ball's frame (the direction into the ball's axes, normalised, times the
// radius), carried back through the basis and measured in the plane's frame: not the floats of centre - n * radius once
// the ball has turned. The normal is the rotation's third column, the distance the point's z in the plane's frame, the point
// on the plane its projection; a point off the finite plane is none. A centre farther than radius + thresh + 0.001 from the
// plane cannot reach it (the arithmetic's rounding is ~1e-6): it skips the rest.
RLMAX_HD Contact sphere_plane(V3 c, const M3& r, float radius, const PlaneXf& x, float thresh) {
    const V3 n = v3(x.r0.z, x.r1.z, x.r2.z);
    Contact out = no_contact();
    if (dot(n, c - x.o) >= radius + thresh + 0.001f) return out;
    const V3 dir = v3(-dot(r.c0, n), -dot(r.c1, n), -dot(r.c2, n));
    const V3 v = (dir * (1.0f / sqrtf(dot(dir, dir)))) * radius;
    // the vertex's coordinate along the plane's column `col`: the column against the ball's axes, plus the centre's offset
    auto along = [&](V3 col) {
        return ((dot(col, r.c0) * v.x + dot(col, r.c1) * v.y) + dot(col, r.c2) * v.z) + (dot(col, c) - dot(col, x.o));
    };
    const float z = along(n);
    if (z >= thresh) return out;
    const float px = along(v3(x.r0.x, x.r1.x, x.r2.x)), py = along(v3(x.r0.y, x.r1.y, x.r2.y));
    const float lx = thresh + x.hx, ly = thresh + x.hy;
    if (px < -lx || py < -ly || lx < px || ly < py) return out;
    out.pb = v3((x.r0.x * px + x.r0.y * py) + x.o.x, (x.r1.x * px + x.r1.y * py) + x.o.y,
                (x.r2.x * px + x.r2.y * py) + x.o.z);
    out.n = n;
    out.d = z;
    out.ok = true;
    return out;
}

// Sphere vs box (sphere_obb_collision_alg.rs). A = sphere, B = box. Contact point on the box surface plus margin.
RLMAX_HD Contact sphere_obb(V3 sc, float radius, const Obb& b, float thresh) {
    Contact out = no_contact();
    V3 local = mul_t(b.r, sc - b.c);
    V3 closest = vclamp(local, -b.h, b.h);
    V3 delta = local - closest;
    float d2 = len2(delta);
    float inter = radius + b.margin;
    float cdist = inter + thresh;
    if (d2 > cdist * cdist) return out;
    float dist;
    if (d2 > FLT_EPS) {
        dist = sqrtf(d2);
        delta = delta * (1.0f / dist);
    } else {
        // deepest face (get_sphere_penetration)
        float md = b.h.x - local.x;
        V3 cl = v3(b.h.x, local.y, local.z), nn = v3(1, 0, 0);
        float f;
        f = b.h.x + local.x; if (f < md) { md = f; cl = v3(-b.h.x, local.y, local.z); nn = v3(-1, 0, 0); }
        f = b.h.y - local.y; if (f < md) { md = f; cl = v3(local.x, b.h.y, local.z); nn = v3(0, 1, 0); }
        f = b.h.y + local.y; if (f < md) { md = f; cl = v3(local.x, -b.h.y, local.z); nn = v3(0, -1, 0); }
        f = b.h.z - local.z; if (f < md) { md = f; cl = v3(local.x, local.y, b.h.z); nn = v3(0, 0, 1); }
        f = b.h.z + local.z; if (f < md) { md = f; cl = v3(local.x, local.y, -b.h.z); nn = v3(0, 0, -1); }
        dist = -md;
        closest = cl;
        delta = nn;
    }
    V3 nb = b.r * delta;
    V3 pbox = b.c + b.r * closest;
    out.n = nb;
    out.pb = pbox + nb * b.margin;
    out.d = dist - inter;
    out.ok = true;
    return out;
}

// Closest point on triangle (a, b, c) to p (Ericson, Real-Time Collision Detection 5.1.5).
RLMAX_HD V3 closest_on_tri(V3 p, V3 a, V3 b, V3 c) {
    V3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    V3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return a + ab * (d1 / (d1 - d3));
    V3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return a + ac * (d2 / (d2 - d6));
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    float denom = 1.0f / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

// Closest points of segments p1q1 and p2q2 (Ericson 5.1.9).
RLMAX_HD void closest_seg_seg(V3 p1, V3 q1, V3 p2, V3 q2, V3* c1, V3* c2) {
    V3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    float s, t;
    if (a <= FLT_EPS && e <= FLT_EPS) {
        s = t = 0.0f;
    } else if (a <= FLT_EPS) {
        s = 0.0f;
        t = clampf(f / e, 0.0f, 1.0f);
    } else {
        float c = dot(d1, r);
        if (e <= FLT_EPS) {
            t = 0.0f;
            s = clampf(-c / a, 0.0f, 1.0f);
        } else {
            float bb = dot(d1, d2), denom = a * e - bb * bb;
            s = denom != 0.0f ? clampf((bb * f - c * e) / denom, 0.0f, 1.0f) : 0.0f;
            t = (bb * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = clampf(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = clampf((bb - c) / a, 0.0f, 1.0f);
            }
        }
    }
    *c1 = p1 + d1 * s;
    *c2 = p2 + d2 * t;
}

// Exact closest points between an axis-aligned box (half extents h, at the origin) and a separated triangle: the
// minimum over triangle vertex vs box, box vertex vs triangle and edge vs edge (the closest pair of two separated
// convex polytopes is always one of these). Returns the distance; *pa on the box, *pb on the triangle.
RLMAX_HD float box_tri_closest(V3 h, const V3 q[3], V3* pa, V3* pb) {
    float best = FLT_BIG;
    auto take = [&](V3 a, V3 b) {
        float d2 = len2(a - b);
        if (d2 < best) { best = d2; *pa = a; *pb = b; }
    };
    for (int i = 0; i < 3; i++) take(vclamp(q[i], -h, h), q[i]);
    for (int k = 0; k < 8; k++) {
        V3 v = v3((k & 1) ? h.x : -h.x, (k & 2) ? h.y : -h.y, (k & 4) ? h.z : -h.z);
        take(v, closest_on_tri(v, q[0], q[1], q[2]));
    }
    for (int ax = 0; ax < 3; ax++) {
        for (int k = 0; k < 4; k++) {
            float u = (k & 1) ? 1.0f : -1.0f, w = (k & 2) ? 1.0f : -1.0f;
            V3 e0, e1;
            if (ax == 0) { e0 = v3(-h.x, u * h.y, w * h.z); e1 = v3(h.x, u * h.y, w * h.z); }
            else if (ax == 1) { e0 = v3(u * h.x, -h.y, w * h.z); e1 = v3(u * h.x, h.y, w * h.z); }
            else { e0 = v3(u * h.x, w * h.y, -h.z); e1 = v3(u * h.x, w * h.y, h.z); }
            for (int i = 0; i < 3; i++) {
                V3 ca, cb;
                closest_seg_seg(e0, e1, q[i], q[(i + 1) % 3], &ca, &cb);
                take(ca, cb);
            }
        }
    }
    return sqrtf(best);
}

// Box vs triangle, one contact per triangle, A = box, B = triangle, with the semantics of the game's convex-triangle
// path (Bullet btGjkPairDetector on the box's implicit shape plus its margin): when the implicit box (half extents
// without margin) is separated from the triangle, the contact is the exact closest-point pair, reported if its
// distance is within margin + breaking threshold, normal along the pair, depth = distance - margin. When the implicit
// box overlaps the triangle (the penetration-solver case) the contact is the separating axis of least penetration of
// the full box: face normal, 3 box axes, 9 edge crosses.
// The game then replaces the normal by the triangle's face normal, (v1 - v0) x (v2 - v0) normalised, with no flip,
// for both the separated and the penetrating result, and keeps the distance and the points (its btGjkPairDetector,
// bin 2a79c0, in the m_fixContactNormalDirection block when B is a triangle; truth/bin/tri_aabb_gate.md).
RLMAX_HD Contact obb_tri(const Obb& b, const Tri& t, const float* goal_ly, float thresh, EpaScratch& sc) {
    Contact out = no_contact();
    if (quirk::GJK_BOX_TRI) {
        // Bullet's GJK distance query, then its penetration solver (gjk.h), in the mesh body's frame (its origin is the
        // triangle transform the detector sees).
        // A contact reaches the manifold only with a computed distance |v| - margin within the breaking threshold
        // (man_add's test, btManifoldResult::addContactPoint), and GJK's |v| is never less than the implicit box's
        // separation along any axis; the penetration paths need |v| below 0.01 BT, under the margin. So a separation
        // beyond margin + threshold (plus a float allowance) along the triangle's normal cannot add a contact, and GJK is
        // skipped with the same outcome. (The box face axes are as sound but measured slower: their test costs more
        // than the GJK calls it saves.)
        if (quirk::GJK_PLANE_REJECT) {
            const float bound = b.margin + thresh + GJK_PLANE_REJECT_ALLOWANCE;
            const float sep = absf(dot(t.n, b.c - t.p0)) -
                              (b.h.x * absf(dot(t.n, b.r.c0)) + b.h.y * absf(dot(t.n, b.r.c1)) + b.h.z * absf(dot(t.n, b.r.c2)));
            if (sep > bound) return out;
        }
        const V3 to = tri_origin(t);
        V3 tv[3];
        if (t.flags & TF_ORIGIN_GOAL) goal_tri_local(t, goal_ly, tv);
        else { tv[0] = t.p0; tv[1] = t.p1; tv[2] = t.p2; }  // the world origin: p - 0
        const float md = b.margin + thresh;
        GjkResult g = gjk_box_tri(b.h, b.margin, b.r, b.c, tv, to, md * md, sc);
        // A failed GJK/EPA query emits no contact; the geometric fallback below is only for the non-GJK path.
        if (!g.valid || !g.contact) return out;
        out.n = t.n;  // the game's triangle-normal rule (see above)
        out.pb = g.pb;
        out.d = g.d;
        out.ok = true;
        return out;
    } else {
        // implicit box separated? exact closest points (in the box frame)
        V3 q[3] = {mul_t(b.r, t.p0 - b.c), mul_t(b.r, t.p1 - b.c), mul_t(b.r, t.p2 - b.c)};
        bool sep = false;
        V3 tn = mul_t(b.r, t.n);
        auto gap = [&](V3 a) {
            float l2 = len2(a);
            if (l2 < 1e-12f) return;
            float rb = b.h.x * absf(a.x) + b.h.y * absf(a.y) + b.h.z * absf(a.z);
            float p0 = dot(q[0], a), p1 = dot(q[1], a), p2 = dot(q[2], a);
            if (fmin_(p0, fmin_(p1, p2)) > rb || fmax_(p0, fmax_(p1, p2)) < -rb) sep = true;
        };
        gap(tn);
        gap(v3(1, 0, 0)); gap(v3(0, 1, 0)); gap(v3(0, 0, 1));
        V3 le[3] = {q[1] - q[0], q[2] - q[1], q[0] - q[2]};
        for (int i = 0; i < 3; i++) {
            gap(cross(v3(1, 0, 0), le[i])); gap(cross(v3(0, 1, 0), le[i])); gap(cross(v3(0, 0, 1), le[i]));
        }
        if (sep) {
            V3 pa, pb;
            float dist = box_tri_closest(b.h, q, &pa, &pb);
            if (!(dist < b.margin + thresh) || dist <= FLT_EPS) return out;
            (void)pa;
            out.n = t.n;  // the game's triangle-normal rule (see above)
            out.pb = b.c + b.r * pb;
            out.d = dist - b.margin;
            out.ok = true;
            return out;
        }
    }
    V3 fh = obb_full_half(b);
    V3 q0 = t.p0 - b.c, q1 = t.p1 - b.c, q2 = t.p2 - b.c;
    float best_pen = FLT_BIG;
    V3 best_axis = t.n;
    bool best_is_face = true;
    int best_kind = 0, best_i = 0, best_j = 0;  // 0 triangle face, 1 box face i, 2 box edge i x triangle edge j
    int kind = 0, ki = 0, kj = 0;
    bool separated = false;
    auto test = [&](V3 axis, bool is_face) {
        float l2 = len2(axis);
        if (l2 < 1e-12f) return;
        V3 a = axis * (1.0f / sqrtf(l2));
        float rb = fh.x * absf(dot(b.r.c0, a)) + fh.y * absf(dot(b.r.c1, a)) + fh.z * absf(dot(b.r.c2, a));
        float p0 = dot(q0, a), p1 = dot(q1, a), p2 = dot(q2, a);
        float tmin = fmin_(p0, fmin_(p1, p2)), tmax = fmax_(p0, fmax_(p1, p2));
        float pen_pos = rb - tmin;  // box leaves towards -a
        float pen_neg = tmax + rb;  // box leaves towards +a
        if (pen_pos < -thresh || pen_neg < -thresh) separated = true;
        float pen = fmin_(pen_pos, pen_neg);
        if (pen < best_pen) {
            best_pen = pen;
            best_axis = pen_pos < pen_neg ? -a : a;  // normal on B pointing to A
            best_is_face = is_face;
            best_kind = kind;
            best_i = ki;
            best_j = kj;
        }
    };
    const V3 bax[3] = {b.r.c0, b.r.c1, b.r.c2};
    const V3 tp[3] = {t.p0, t.p1, t.p2};
    kind = 0;
    test(t.n, true);
    kind = 1;
    for (ki = 0; ki < 3; ki++) test(bax[ki], false);
    kind = 2;
    for (ki = 0; ki < 3; ki++)
        for (kj = 0; kj < 3; kj++) test(cross(bax[ki], tp[(kj + 1) % 3] - tp[kj]), false);
    if (separated) return out;
    if (best_pen < 0.0f && !best_is_face) return out;
    // Witnesses on the winning feature pair, as the penetration solver (EPA) reports them: the point on B lies on the
    // triangle's own feature, so a contact on an internal edge is seen there by the internal-edge adjustment.
    //  - triangle face: the box's deepest vertex along -axis, projected onto the triangle's plane;
    //  - box face: the triangle's deepest point along +axis (the mean of the vertices tied there);
    //  - edge pair: the closest points of the box's deepest edge along -axis and the triangle edge.
    V3 la = mul_t(b.r, best_axis);
    V3 sgn = v3(la.x > 0.0f ? -1.0f : 1.0f, la.y > 0.0f ? -1.0f : 1.0f, la.z > 0.0f ? -1.0f : 1.0f);
    V3 sup = b.c + b.r * (sgn * fh);
    V3 pb;
    if (best_kind == 0) {
        pb = sup + best_axis * best_pen;
    } else if (best_kind == 1) {
        float top = fmax_(dot(tp[0], best_axis), fmax_(dot(tp[1], best_axis), dot(tp[2], best_axis)));
        const float tie = 1e-4f;
        V3 sum = v3s(0.0f);
        float cnt = 0.0f;
        for (int v = 0; v < 3; v++)
            if (dot(tp[v], best_axis) >= top - tie) {
                sum += tp[v];
                cnt += 1.0f;
            }
        pb = sum * (1.0f / cnt);
    } else {
        const float hi = vget(fh, best_i);
        V3 mid = sup - bax[best_i] * (vget(sgn, best_i) * hi);
        V3 ca, cb;
        closest_seg_seg(mid - bax[best_i] * hi, mid + bax[best_i] * hi, tp[best_j], tp[(best_j + 1) % 3], &ca, &cb);
        pb = cb;
    }
    out.n = t.n;  // the game's triangle-normal rule (see above)
    out.d = -best_pen;
    out.pb = pb;
    out.ok = true;
    return out;
}

// ---- rays: fraction in [0, best) or no hit -------------------

struct RayHit {
    float frac;
    V3 n;
    bool ok;
};

// btTriangleRaycastCallback::processTriangle: the normal is the raw edge cross product and the edge tolerance scales with its
// square, so the test is relative to the triangle's size.
RLMAX_HD void ray_tri(V3 from, V3 to, const Tri& t, V3 nraw, RayHit* best) {
    const float dist = dot(t.p0, nraw);
    const float da = dot(nraw, from) - dist;
    const float db = dot(nraw, to) - dist;
    if (da * db >= 0.0f) return;
    const float proj = da - db;
    if (proj == 0.0f) return;
    const float frac = da / proj;
    if (frac >= best->frac) return;
    const float tol = -0.0001f * len2(nraw);
    const float rest = 1.0f - frac;
    const V3 p = v3(rest * from.x + frac * to.x, rest * from.y + frac * to.y, rest * from.z + frac * to.z);
    const V3 v0p = t.p0 - p, v1p = t.p1 - p, v2p = t.p2 - p;
    if (dot(cross(v0p, v1p), nraw) < tol) return;
    if (dot(cross(v1p, v2p), nraw) < tol) return;
    if (dot(cross(v2p, v0p), nraw) < tol) return;
    const V3 n = nraw * (1.0f / sqrtf(len2(nraw)));
    best->frac = frac;
    best->n = da <= 0.0f ? -n : n;
    best->ok = true;
}
// The game's ray walks the mesh's BVH and tests only the triangles whose leaf box it crosses, so a ray that catches a triangle within
// the edge tolerance but misses its leaf box doesn't hit it (btQuantizedBvh::walkStacklessQuantizedTreeAgainstRay, btRayAabb2).
// box: min, max, in the frame of from and to.
RLMAX_HD bool ray_leaf_box(V3 from, V3 to, const float* box) {
    const V3 d = to - from;
    const float r = 1.0f / sqrtf(d.y * d.y + d.x * d.x + d.z * d.z);
    const V3 u = v3(d.x * r, d.y * r, d.z * r);
    const float lambda_max = u.x * d.x + u.y * d.y + u.z * d.z;
    const V3 inv = v3(u.x == 0.0f ? 1e18f : 1.0f / u.x, u.y == 0.0f ? 1e18f : 1.0f / u.y, u.z == 0.0f ? 1e18f : 1.0f / u.z);
    const float tmin = ((inv.x < 0.0f ? box[3] : box[0]) - from.x) * inv.x, tmax = ((inv.x < 0.0f ? box[0] : box[3]) - from.x) * inv.x;
    const float tymin = ((inv.y < 0.0f ? box[4] : box[1]) - from.y) * inv.y, tymax = ((inv.y < 0.0f ? box[1] : box[4]) - from.y) * inv.y;
    if (!(tmin <= tymax) || !(tymin <= tmax)) return false;
    const float t0 = tymin <= tmin ? tmin : tymin, t1 = tmax <= tymax ? tmax : tymax;
    const float tzmin = ((inv.z < 0.0f ? box[5] : box[2]) - from.z) * inv.z, tzmax = ((inv.z < 0.0f ? box[2] : box[5]) - from.z) * inv.z;
    if (!(t0 <= tzmax) || !(tzmin <= t1)) return false;
    return (tzmin <= t0 ? t0 : tzmin) < lambda_max && 0.0f < (t1 <= tzmax ? t1 : tzmax);
}
RLMAX_HD V3 tri_raw_normal(const Tri& t) { return cross(t.p1 - t.p0, t.p2 - t.p0); }
RLMAX_HD void ray_tri(V3 from, V3 to, const Tri& t, RayHit* best) { ray_tri(from, to, t, tri_raw_normal(t), best); }

// btStaticPlaneShape as a ray sees it: hits only from the front, the game's origin and normal
RLMAX_HD void ray_plane_game(V3 from, V3 to, V3 origin, V3 n, RayHit* best) {
    constexpr float EPS = 1.1920929e-07f;
    const V3 delta = to - from;
    const V3 offset = origin - from;
    const float denom = -(delta.y * n.y + delta.x * n.x + delta.z * n.z);
    const float numer = -(offset.y * n.y + offset.x * n.x + offset.z * n.z);
    if (!(EPS < denom && EPS < numer && numer <= denom)) return;
    const float frac = numer / denom;
    if (!(frac < best->frac)) return;
    best->frac = frac;
    best->n = n;
    best->ok = true;
}

// The wheel ray against the ball as the game's btCollisionWorld::rayTestSingle makes it: btGjkConvexCast of a point
// against the sphere. GJK sees the sphere as its centre plus
// a margin of one radius, so the distance is |p - c| - r with normal (p - c)/|p - c|. Conservative advancement stops once
// the point is within 0.001 BT of the surface (at most 64 steps), so the fraction and normal are those of that point,
// not the exact root. A start inside the ball hits at fraction 0 when the ray heads inward and misses when it heads
// outward; a start at the centre finds no axis: no hit. The reject test is the game's (ball farther than radius + ray
// length + 1e-4 from the start).
RLMAX_HD bool sphere_surface_dist(V3 p, V3 c, float radius, float* dist, V3* n) {
    const V3 offset = (p + c) * 0.5f;  // the cast's GJK takes both points about their midpoint
    const V3 w = (p - offset) - (c - offset);
    const float l2 = dot(w, w);
    if (!(l2 > 1.1920929e-7f * 1.1920929e-7f)) return false;
    const float rl = 1.0f / sqrtf(l2);
    *n = w * rl;
    *dist = (1.0f / rl) - radius;
    return true;
}

RLMAX_HD void ray_sphere(V3 from, V3 to, V3 c, float radius, RayHit* best) {
    const V3 r = to - from;
    {
        const V3 dc = c - from;
        const float rej = radius + sqrtf(dot(r, r)) + 1.0e-4f;
        if (dot(dc, dc) > rej * rej) return;
    }
    float dist = 0.0f;
    V3 n = v3s(0.0f);
    if (!sphere_surface_dist(from, c, radius, &dist, &n)) return;
    float lambda = 0.0f, last = 0.0f;
    bool penetrated = false;
    int iter = 0;
    while (dist > 0.001f) {
        if (++iter > 64) return;
        lambda = lambda - dist / dot(r, n);
        if (lambda > 1.0f || lambda < 0.0f || lambda <= last) return;
        last = lambda;
        float next = 0.0f;
        if (!sphere_surface_dist(from * (1.0f - lambda) + to * lambda, c * lambda + c * (1.0f - lambda), radius, &next, &n)) return;
        if (next < 0.0f) { penetrated = true; break; }
        dist = next;
    }
    if (!penetrated && dot(n, r) >= 0.0f) return;
    if (lambda >= best->frac) return;
    best->frac = lambda;
    best->n = normalize_or_zero(n);  // the game normalises the cast's normal once more
    best->ok = true;
}

// A point against the box with its margin, by Bullet's GJK (btGjkPairDetector on a point and a box, as the game's
// convex cast calls it): the distance to the margined surface and the unit normal
// from the box to the point. Both are only what the simplex has converged to, which is why the cast's hit normal is not
// the exact surface normal. `dist` keeps the smallest distance seen, as the game's result does across a cast.
RLMAX_HD bool gjk_point_box(V3 pa, const Obb& b, bool& has, float& dist, V3& n) {
    const V3 offset = (pa + b.c) * 0.5f;
    const V3 la = pa - offset, lb = b.c - offset;
    V3 axis = v3(0.0f, 1.0f, 0.0f);
    float sq = 1.0e18f;
    GjkSimplex sx;
    gjk_reset(sx);
    int iter = 0;
    for (;;) {
        const V3 sep = v3(dot(b.r.c0, axis), dot(b.r.c1, axis), dot(b.r.c2, axis));
        const V3 ql = gjk_box_support(b.h, sep);
        const V3 qw = v3((b.r.c0.x * ql.x + b.r.c1.x * ql.y) + b.r.c2.x * ql.z + lb.x,
                         (b.r.c0.y * ql.x + b.r.c1.y * ql.y) + b.r.c2.y * ql.z + lb.y,
                         (b.r.c0.z * ql.x + b.r.c1.z * ql.y) + b.r.c2.z * ql.z + lb.z);
        const V3 w = la - qw;
        const float delta = dot(axis, w);
        if (delta > 0.0f && delta * delta > sq * 1.0e18f) break;
        if (gjk_in_simplex(sx, w)) break;
        if (sq - delta <= sq * 1.0e-6f) break;
        gjk_add(sx, w, la, qw);
        if (!gjk_update(sx)) break;
        const V3 na = sx.cv;
        if (len2(na) < 1.0e-6f) {
            axis = na;
            break;
        }
        const float prev = sq;
        sq = len2(na);
        if (prev - sq <= 1.1920929e-7f * prev) break;
        axis = na;
        if (iter++ > GJK_MAX_ITER || sx.n == 4) break;
    }
    gjk_update(sx);
    const float l2 = len2(axis);
    if (!(l2 > 1.1920929e-7f * 1.1920929e-7f)) return has;
    const float rl = 1.0f / sqrtf(l2);
    V3 nrm = axis * rl;
    const float d = (1.0f / rl) - b.margin;
    if (!(d < 0.0f || d * d < 1.0e18f)) return has;
    const V3 hm = b.h + v3s(b.margin);
    const V3 ext = v3((hm.x * absf(b.r.c0.x) + hm.y * absf(b.r.c1.x)) + hm.z * absf(b.r.c2.x),
                      (hm.x * absf(b.r.c0.y) + hm.y * absf(b.r.c1.y)) + hm.z * absf(b.r.c2.y),
                      (hm.x * absf(b.r.c0.z) + hm.y * absf(b.r.c1.z)) + hm.z * absf(b.r.c2.z));
    const V3 pb = ((lb - ext) + (lb + ext)) * 0.5f;
    if (dot(la - pb, nrm) < 0.0f) nrm = nrm * -1.0f;
    if (d < dist) {
        has = true;
        n = nrm;
        dist = d;
    }
    return has;
}

// The wheel ray against a car's box: the game's GJK convex cast of a point along the ray (btGjkConvexCast, radius 0.001),
// the box not moving; a ray that starts inside the margin, or heads away, hits nothing. The narrowed body-overlap
// gate makes this cast rare, so keep its GJK loop out of the common wheel-ray code.
RLMAX_COLD void ray_obb(V3 from, V3 to, const Obb& b, RayHit* best) {
    const V3 r = to - from;
    bool has = false;
    float dist = 1.0e18f;
    V3 n = v3s(0.0f);
    if (!gjk_point_box(from, b, has, dist, n)) return;
    float lambda = 0.0f;
    int iter = 0;
    for (;;) {
        if (dist <= 0.001f) {
            if (-0.0f <= dot(r, n)) return;
            break;
        }
        if (++iter > 64) return;
        const float next = lambda - dist / dot(r, n);
        if (next > 1.0f || next < 0.0f || next <= lambda) return;
        const float s = 1.0f - next;
        const Obb at{v3(next * b.c.x + s * b.c.x, next * b.c.y + s * b.c.y, next * b.c.z + s * b.c.z), b.r, b.h, b.margin};
        if (!gjk_point_box(v3(next * to.x + s * from.x, next * to.y + s * from.y, next * to.z + s * from.z), at, has, dist, n)) return;
        lambda = next;
        if (!(0.0f <= dist)) break;
    }
    const float nl2 = len2(n);
    if (!(nl2 > 1.0e-4f) || !(lambda < 1.0f)) return;
    if (lambda >= best->frac) return;
    best->frac = lambda;
    best->n = n * (1.0f / sqrtf(nl2));
    best->ok = true;
}

// Keep the deepest `K` contacts in fixed slots (used by the debug hooks).
template <int K>
RLMAX_HD void keep_deepest(Contact* slots, const Contact& c) {
    if (!c.ok) return;
    Contact cur = c;
#pragma unroll
    for (int i = 0; i < K; i++) {
        bool take = !slots[i].ok || cur.d < slots[i].d;
        Contact tmp = slots[i];
        if (take) {
            slots[i] = cur;
            cur = tmp;
        }
        if (!cur.ok) break;
    }
}

// Box vs box: Bullet's btBoxBoxDetector (ODE dBoxBox2) on the full boxes (half extents with margin). The 15-axis test
// picks one normal for the pair (face axes first; an edge axis wins only if 1.05 x its separation beats the best face,
// ODE's fudge factor, with 1e-5 added to |R| for the edge axes). Face axis: the other box's incident face is clipped
// against the reference face rectangle (intersectRectQuad2), points below the reference face kept, at most K by
// cullPoints2 (deepest first, then by angle round the centroid). Edge axis: one point, the closest approach of the
// two edges. Overlap only. A = first box, B = second; normals on B point towards A, points on B.
RLMAX_HD float obb_radius(const Obb& b, V3 a) {
    V3 fh = obb_full_half(b);
    return fh.x * absf(dot(b.r.c0, a)) + fh.y * absf(dot(b.r.c1, a)) + fh.z * absf(dot(b.r.c2, a));
}

// ODE's intersectRectQuad2: clip the quad p (4 points, xy pairs) against the rectangle |x| <= h[0], |y| <= h[1];
// returns the number of points (at most 8) in ret.
RLMAX_HD int intersect_rect_quad2(const float h[2], const float p[8], float ret[16]) {
    int nq = 4, nr = 0;
    float buf_a[16], buf_b[16];
    for (int i = 0; i < 8; i++) buf_a[i] = p[i];
    float* q = buf_a;
    float* r = buf_b;
    for (int dir = 0; dir <= 1; dir++) {
        for (int sign = -1; sign <= 1; sign += 2) {
            const float sg = (float)sign;
            float* pq = q;
            float* pr = r;
            nr = 0;
            for (int i = nq; i > 0; i--) {
                if (sg * pq[dir] < h[dir]) {
                    pr[0] = pq[0];
                    pr[1] = pq[1];
                    pr += 2;
                    nr++;
                    if (nr & 8) { q = r; goto done; }
                }
                float* nextq = (i > 1) ? pq + 2 : q;
                if ((sg * pq[dir] < h[dir]) != (sg * nextq[dir] < h[dir])) {
                    pr[1 - dir] = pq[1 - dir] + (nextq[1 - dir] - pq[1 - dir]) / (nextq[dir] - pq[dir]) * (sg * h[dir] - pq[dir]);
                    pr[dir] = sg * h[dir];
                    pr += 2;
                    nr++;
                    if (nr & 8) { q = r; goto done; }
                }
                pq += 2;
            }
            float* t = q;
            q = r;
            r = t;
            nq = nr;
        }
    }
done:
    for (int i = 0; i < nr * 2; i++) ret[i] = q[i];
    return nr;
}

// ODE's cullPoints2: keep m of the n polygon points p, starting from i0, then the points whose angle round the
// centroid is nearest i0's plus j * 2 pi / m.
RLMAX_HD void cull_points2(int n, const float p[], int m, int i0, int iret[]) {
    float a, cx, cy, q;
    if (n == 1) {
        cx = p[0];
        cy = p[1];
    } else if (n == 2) {
        cx = 0.5f * (p[0] + p[2]);
        cy = 0.5f * (p[1] + p[3]);
    } else {
        a = 0.0f;
        cx = 0.0f;
        cy = 0.0f;
        for (int i = 0; i < n - 1; i++) {
            q = p[i * 2] * p[i * 2 + 3] - p[i * 2 + 2] * p[i * 2 + 1];
            a += q;
            cx += q * (p[i * 2] + p[i * 2 + 2]);
            cy += q * (p[i * 2 + 1] + p[i * 2 + 3]);
        }
        q = p[n * 2 - 2] * p[1] - p[0] * p[n * 2 - 1];
        if (absf(a + q) > FLT_EPS) a = 1.0f / (3.0f * (a + q));
        else a = FLT_BIG;
        cx = a * (cx + q * (p[n * 2 - 2] + p[0]));
        cy = a * (cy + q * (p[n * 2 - 1] + p[1]));
    }
    float ang[8];
    for (int i = 0; i < n; i++) ang[i] = game_atan2f(p[i * 2 + 1] - cy, p[i * 2] - cx);
    bool avail[8];
    for (int i = 0; i < n; i++) avail[i] = true;
    avail[i0] = false;
    iret[0] = i0;
    for (int j = 1; j < m; j++) {
        a = (float)j * (2.0f * PI / (float)m) + ang[i0];
        if (a > PI) a -= 2.0f * PI;
        float maxdiff = 1e9f;
        iret[j] = i0;
        for (int i = 0; i < n; i++) {
            if (!avail[i]) continue;
            float diff = absf(ang[i] - a);
            if (diff > PI) diff = 2.0f * PI - diff;
            if (diff < maxdiff) {
                maxdiff = diff;
                iret[j] = i;
            }
        }
        avail[iret[j]] = false;
    }
}

// Box vs box (ODE dBoxBox as the game runs it). The pair's A is the game's second box: the axes, the offset and the
// signs are taken from B as box 1, and the contact comes back as
// point on B, normal on B towards A, distance: B's point is A's moved along the normal by the depth.
template <int K>
// Car pairs reaching narrow phase are rare; keep clipping code out of the tick entry.
RLMAX_COLD void obb_obb(const Obb& pair_a, const Obb& pair_b, float thresh, Contact out[K], V3* point_a = nullptr) {
    (void)thresh;
    const Obb& a = pair_b;
    const Obb& b = pair_a;
#pragma unroll
    for (int i = 0; i < K; i++) out[i] = no_contact();
    const V3 Ah = obb_full_half(a), Bh = obb_full_half(b);
    const float A[3] = {Ah.x, Ah.y, Ah.z}, B[3] = {Bh.x, Bh.y, Bh.z};
    const V3 R1[3] = {a.r.c0, a.r.c1, a.r.c2}, R2[3] = {b.r.c0, b.r.c1, b.r.c2};
    const V3 p = b.c - a.c;
    const float pp[3] = {dot(R1[0], p), dot(R1[1], p), dot(R1[2], p)};
    float R[3][3], Q[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            R[i][j] = dot(R1[i], R2[j]);
            Q[i][j] = absf(R[i][j]);
        }
    float s = -FLT_BIG;
    bool invert = false, normal_r = false;
    int code = 0;
    V3 normal = v3s(0.0f), normal_c = v3s(0.0f);
    // face axes of A, then of B
    for (int i = 0; i < 3; i++) {
        float e1 = pp[i];
        float e2 = ((A[i] + B[0] * Q[i][0]) + B[1] * Q[i][1]) + B[2] * Q[i][2];
        float s2 = absf(e1) - e2;
        if (s2 > 0.0f) return;
        if (s2 > s) { s = s2; normal = R1[i]; normal_r = true; invert = e1 < 0.0f; code = i + 1; }
    }
    for (int j = 0; j < 3; j++) {
        float e1 = dot(R2[j], p);
        float e2 = ((A[0] * Q[0][j] + A[1] * Q[1][j]) + A[2] * Q[2][j]) + B[j];
        float s2 = absf(e1) - e2;
        if (s2 > 0.0f) return;
        if (s2 > s) { s = s2; normal = R2[j]; normal_r = true; invert = e1 < 0.0f; code = 4 + j; }
    }
    // edge axes u_i x v_j, |R| fudged
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) Q[i][j] += 1e-5f;
    bool separated = false;
    auto tst = [&](float e1, float e2, float n1, float n2, float n3, int cc) {
        if (separated) return;
        float s2 = absf(e1) - e2;
        if (s2 > FLT_EPS) { separated = true; return; }
        float l = sqrtf(n1 * n1 + n2 * n2 + n3 * n3);
        if (l > FLT_EPS) {
            s2 /= l;
            if (s2 * 1.05f > s) {
                s = s2; normal_r = false; normal_c = v3(n1 / l, n2 / l, n3 / l); invert = e1 < 0.0f; code = cc;
            }
        }
    };
    // ODE's R_ij = u_i . v_j (1-based): R[i-1][j-1]
    const float R11 = R[0][0], R12 = R[0][1], R13 = R[0][2], R21 = R[1][0], R22 = R[1][1], R23 = R[1][2];
    const float R31 = R[2][0], R32 = R[2][1], R33 = R[2][2];
    const float Q11 = Q[0][0], Q12 = Q[0][1], Q13 = Q[0][2], Q21 = Q[1][0], Q22 = Q[1][1], Q23 = Q[1][2];
    const float Q31 = Q[2][0], Q32 = Q[2][1], Q33 = Q[2][2];
    tst(pp[2] * R21 - pp[1] * R31, A[1] * Q31 + A[2] * Q21 + B[1] * Q13 + B[2] * Q12, 0.0f, -R31, R21, 7);
    tst(pp[2] * R22 - pp[1] * R32, A[1] * Q32 + A[2] * Q22 + B[0] * Q13 + B[2] * Q11, 0.0f, -R32, R22, 8);
    tst(pp[2] * R23 - pp[1] * R33, A[1] * Q33 + A[2] * Q23 + B[0] * Q12 + B[1] * Q11, 0.0f, -R33, R23, 9);
    tst(pp[0] * R31 - pp[2] * R11, A[0] * Q31 + A[2] * Q11 + B[1] * Q23 + B[2] * Q22, R31, 0.0f, -R11, 10);
    tst(pp[0] * R32 - pp[2] * R12, A[0] * Q32 + A[2] * Q12 + B[0] * Q23 + B[2] * Q21, R32, 0.0f, -R12, 11);
    tst(pp[0] * R33 - pp[2] * R13, A[0] * Q33 + A[2] * Q13 + B[0] * Q22 + B[1] * Q21, R33, 0.0f, -R13, 12);
    tst(pp[1] * R11 - pp[0] * R21, A[0] * Q21 + A[1] * Q11 + B[1] * Q33 + B[2] * Q32, -R21, R11, 0.0f, 13);
    tst(pp[1] * R12 - pp[0] * R22, A[0] * Q22 + A[1] * Q12 + B[0] * Q33 + B[2] * Q31, -R22, R12, 0.0f, 14);
    tst(pp[1] * R13 - pp[0] * R23, A[0] * Q23 + A[1] * Q13 + B[0] * Q32 + B[1] * Q31, -R23, R13, 0.0f, 15);
    if (separated || code == 0) return;
    if (!normal_r) normal = R1[0] * normal_c.x + R1[1] * normal_c.y + R1[2] * normal_c.z;
    if (invert) normal = -normal;
    const float depth = -s;

    if (code > 6) {
        // edge-edge: the closest approach of the two touching edges (dLineClosestApproach); the point is on B's edge
        V3 pa = a.c, pb = b.c;
        for (int j = 0; j < 3; j++) pa += R1[j] * ((dot(normal, R1[j]) > 0.0f ? 1.0f : -1.0f) * A[j]);
        for (int j = 0; j < 3; j++) pb += R2[j] * ((dot(normal, R2[j]) > 0.0f ? -1.0f : 1.0f) * B[j]);
        const V3 ua = R1[(code - 7) / 3], ub = R2[(code - 7) % 3];
        const V3 d = pb - pa;
        const float uaub = dot(ua, ub), q1 = dot(ua, d), q2 = -dot(ub, d);
        float den = 1.0f - uaub * uaub, beta = 0.0f;
        if (den > 0.0001f) {
            den = 1.0f / den;
            beta = (uaub * q1 + q2) * den;
        }
        pb += ub * beta;
        if (point_a) point_a[0] = pb;
        out[0].pb = pb + normal * depth;
        out[0].n = normal;
        out[0].d = -depth;
        out[0].ok = true;
        return;
    }

    // face: reference face of box `a` (code 1..3) or `b` (4..6), incident face of the other
    const V3* Ra = code <= 3 ? R1 : R2;
    const V3* Rb = code <= 3 ? R2 : R1;
    const V3 pa = code <= 3 ? a.c : b.c, pb = code <= 3 ? b.c : a.c;
    const float* Sa = code <= 3 ? A : B;
    const float* Sb = code <= 3 ? B : A;
    const V3 normal2 = code <= 3 ? normal : -normal;
    const float nr[3] = {dot(Rb[0], normal2), dot(Rb[1], normal2), dot(Rb[2], normal2)};
    const float anr[3] = {absf(nr[0]), absf(nr[1]), absf(nr[2])};
    int lanr, a1, a2;
    if (anr[1] > anr[0]) {
        if (anr[1] > anr[2]) { a1 = 0; lanr = 1; a2 = 2; } else { a1 = 0; a2 = 1; lanr = 2; }
    } else {
        if (anr[0] > anr[2]) { lanr = 0; a1 = 1; a2 = 2; } else { a1 = 0; a2 = 1; lanr = 2; }
    }
    V3 center = pb - pa;
    if (nr[lanr] < 0.0f) center += Rb[lanr] * Sb[lanr];
    else center -= Rb[lanr] * Sb[lanr];
    const int code_n = code <= 3 ? code - 1 : code - 4;
    const int code1 = code_n == 0 ? 1 : 0, code2 = code_n == 2 ? 1 : 2;
    const float c1 = dot(center, Ra[code1]), c2 = dot(center, Ra[code2]);
    float m11 = dot(Ra[code1], Rb[a1]), m12 = dot(Ra[code1], Rb[a2]);
    float m21 = dot(Ra[code2], Rb[a1]), m22 = dot(Ra[code2], Rb[a2]);
    const float k1 = m11 * Sb[a1], k2 = m21 * Sb[a1], k3 = m12 * Sb[a2], k4 = m22 * Sb[a2];
    const float quad[8] = {c1 - k1 - k3, c2 - k2 - k4, c1 - k1 + k3, c2 - k2 + k4,
                           c1 + k1 + k3, c2 + k2 + k4, c1 + k1 - k3, c2 + k2 - k4};
    const float rect[2] = {Sa[code1], Sa[code2]};
    float ret[16];
    const int n = intersect_rect_quad2(rect, quad, ret);
    const float det1 = 1.0f / (m11 * m22 - m12 * m21);
    m11 *= det1;
    m12 *= det1;
    m21 *= det1;
    m22 *= det1;
    V3 point[8];
    float dep[8];
    int cnum = 0;
    for (int j = 0; j < n; j++) {
        const float k1j = m22 * (ret[j * 2] - c1) - m12 * (ret[j * 2 + 1] - c2);
        const float k2j = -m21 * (ret[j * 2] - c1) + m11 * (ret[j * 2 + 1] - c2);
        point[cnum] = center + Rb[a1] * k1j + Rb[a2] * k2j;
        dep[cnum] = Sa[code_n] - dot(normal2, point[cnum]);
        if (dep[cnum] >= 0.0f) {
            ret[cnum * 2] = ret[j * 2];
            ret[cnum * 2 + 1] = ret[j * 2 + 1];
            cnum++;
        }
    }
    if (cnum < 1) return;
    const int maxc = K < 4 ? K : 4;
    int sel[8];
    int ns;
    if (cnum <= maxc) {
        for (int j = 0; j < cnum; j++) sel[j] = j;
        ns = cnum;
    } else {
        int i1 = 0;
        float maxdepth = dep[0];
        for (int i = 1; i < cnum; i++)
            if (dep[i] > maxdepth) {
                maxdepth = dep[i];
                i1 = i;
            }
        cull_points2(cnum, ret, maxc, i1, sel);
        ns = maxc;
    }
    for (int q = 0; q < ns; q++) {
        const int j = sel[q];
        V3 pw = point[j] + pa;
        if (code > 3) pw -= normal * dep[j];
        if (point_a) point_a[q] = pw;
        out[q].pb = pw + normal * dep[j];
        out[q].n = normal;
        out[q].d = -dep[j];
        out[q].ok = true;
    }
}

}  // namespace rlmax
