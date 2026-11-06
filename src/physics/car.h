// RLMAX car model. Behaviour as the game has it,
// constants from consts.h. Written over RLMAX's own state and body structs.
#pragma once

#include "body.h"
#include "contacts.h"
#include "state.h"
#ifdef RLMAX_CHECK_RAYREJECT
#include <cstdio>
#endif

namespace rlmax {

// ---- static car geometry (per body preset), Bullet units --------------------------------------------------------------

struct CarGeo {
    V3 hitbox_half;  // implicit half extents (without margin)
    float hitbox_len;  // len(hitbox_half), for the wheel rays' reach reject
    float margin;
    V3 hitbox_off;
    V3 wheel_conn[4];   // chassis-space ray start per wheel (0,1 front; even = left: y negated)
    float rest_len[4];  // suspension rest length minus max travel
    float radius[4];
    float ray_len[4];   // rest + travel + radius
    float force_scale[4];
    V3 inv_inertia_local;
    float contact_threshold;
    // steering curves (degrees vs |forward speed|) with the knots the game's init rewrite leaves (consts.h steer_knots_deg);
    // both front wheels then use the one angle
    Curve<6> steer;
    Curve<2> steer_powerslide;
};

RLMAX_HD CarGeo body_geo(uint32_t preset) {
    const BodyPreset& p = body_preset(preset);
    CarGeo g;
    V3 half = body_half_bt(p);
    float margin = fmin_(0.1f * fmin_(half.x, fmin_(half.y, half.z)), c::CONVEX_DISTANCE_MARGIN);
    g.margin = margin;
    g.hitbox_half = half - v3s(margin);
    g.hitbox_len = len(g.hitbox_half);
    g.hitbox_off = v3(p.off_x, 0.0f, p.off_z) * c::UU_TO_BT;
    // box inertia from the full box: l = 2 * (implicit + margin)
    V3 l = (g.hitbox_half + v3s(margin)) * 2.0f;
    float m = c::CAR_MASS_BT;
    V3 inertia = v3(m / 12.0f * (l.y * l.y + l.z * l.z), m / 12.0f * (l.x * l.x + l.z * l.z),
                    m / 12.0f * (l.x * l.x + l.y * l.y));
    g.inv_inertia_local = v3(1.0f / inertia.x, 1.0f / inertia.y, 1.0f / inertia.z);
    // suspension strengths (Car::suspension_strengths, positive axle distances)
    float front_x = p.front_x * c::UU_TO_BT, rear_x = p.back_x * c::UU_TO_BT;
    float mass = 1.0f / (1.0f / m);
    float front_dist = front_x, rear_dist = -rear_x;
    float total = rear_dist + front_dist;
    float front_frac = front_dist / total;
    float front_axle = front_frac * mass;
    float rear_axle = mass - front_axle;
    float front_wheel = rear_axle * 0.5f, rear_wheel = front_axle * 0.5f;
    float travel = c::MAX_SUSPENSION_TRAVEL_UU * c::UU_TO_BT;
    for (int i = 0; i < 4; i++) {
        bool front = i < 2, left = (i % 2) == 0;
        V3 off = front ? v3(p.front_x, p.front_y, p.front_z) : v3(p.back_x, p.back_y, p.back_z);
        if (left) off.y = -off.y;
        g.wheel_conn[i] = off * c::UU_TO_BT;
        // the game's rest length is measured to the wheel bone, whose z is rounded to 0.01 uu
        const float published_rest = (front ? p.front_rest : p.back_rest) - c::MAX_SUSPENSION_TRAVEL_UU;
        const float bone_z = floorf((off.z - published_rest) * 100.0f + 0.5f) / 100.0f;
        float rest = off.z - bone_z;
        g.rest_len[i] = rest * c::UU_TO_BT;
        g.radius[i] = (front ? p.front_rad : p.back_rad) * c::UU_TO_BT;
        g.ray_len[i] = (g.rest_len[i] + g.radius[i]) + travel;
        g.force_scale[i] = front ? front_wheel : rear_wheel;
    }
    // contact breaking threshold: 0.02 * (bounding sphere of the compound's local AABB + |centre|)
    V3 fh = g.hitbox_half + v3s(margin);
    g.contact_threshold = (len(fh) + len(g.hitbox_off)) * c::CONTACT_BREAKING_THRESHOLD;
    const SteerKnots& k = steer_knots_deg(preset);
    g.steer = curves::STEER_ANGLE_DEG;
    for (int i = 0; i < 6; i++) g.steer.y[i] = k.steer[i];
    g.steer_powerslide = curves::POWERSLIDE_STEER_ANGLE_DEG;
    for (int i = 0; i < 2; i++) g.steer_powerslide.y[i] = k.powerslide[i];
    return g;
}

RLMAX_HD Obb car_obb(const Body& b, const CarGeo& g) {
    return Obb{b.pos + b.r * g.hitbox_off, b.r, g.hitbox_half, g.margin};
}

// Reject rays that miss the full box before the rounded-box GJK cast. The cast's distance tolerance alone can report
// a hit just outside a box edge; the game's local-space ray/box test excludes those rays first.
RLMAX_HD bool ray_car_box_overlap(V3 from, V3 to, const Body& b, const CarGeo& g) {
    V3 a = from - b.pos, z = to - b.pos;
    V3 lf = v3(dot(a, b.r.c0), dot(a, b.r.c1), dot(a, b.r.c2));
    V3 lt = v3(dot(z, b.r.c0), dot(z, b.r.c1), dot(z, b.r.c2));
    V3 ext = g.hitbox_half + v3s(g.margin), lo = g.hitbox_off - ext, hi = g.hitbox_off + ext;
    if ((lf.x < lo.x && lt.x < lo.x) || (lf.x > hi.x && lt.x > hi.x) ||
        (lf.y < lo.y && lt.y < lo.y) || (lf.y > hi.y && lt.y > hi.y) ||
        (lf.z < lo.z && lt.z < lo.z) || (lf.z > hi.z && lt.z > hi.z)) return false;
    V3 delta = lt - lf;
    float length = len(delta);
    if (!(length > 0.0f)) return false;
    V3 dir = delta * (1.0f / length);
    float limit = dot(dir, delta);
    const float d[3] = {dir.x, dir.y, dir.z}, f[3] = {lf.x, lf.y, lf.z};
    const float mn[3] = {lo.x, lo.y, lo.z}, mx[3] = {hi.x, hi.y, hi.z};
    float inv = d[0] == 0.0f ? 1.0e30f : 1.0f / d[0];
    float tmin = ((inv < 0.0f ? mx[0] : mn[0]) - f[0]) * inv;
    float tmax = ((inv < 0.0f ? mn[0] : mx[0]) - f[0]) * inv;
    for (int i = 1; i < 3; i++) {
        inv = d[i] == 0.0f ? 1.0e30f : 1.0f / d[i];
        float t0 = ((inv < 0.0f ? mx[i] : mn[i]) - f[i]) * inv;
        float t1 = ((inv < 0.0f ? mn[i] : mx[i]) - f[i]) * inv;
        if (tmin > t1 || t0 > tmax) return false;
        if (t0 > tmin) tmin = t0;
        if (t1 < tmax) tmax = t1;
    }
    return tmin < limit && tmax > 0.0f;
}

// ---- the world as wheel rays see it --------------------------------------------------------------------------

constexpr int NUM_PLANES = 4;
// floor, ceiling, side walls; plane: n . p = d (BT)
RLMAX_HD void arena_plane(int i, V3* n, float* d) {
    const float X = c::ARENA_X_UU * c::UU_TO_BT, Z = c::ARENA_Z_UU * c::UU_TO_BT;
    if (i == 0) { *n = v3(0, 0, 1); *d = 0.0f; }
    else if (i == 1) { *n = v3(0, 0, -1); *d = -Z; }
    else if (i == 2) { *n = v3(1, 0, 0); *d = -X; }
    else { *n = v3(-1, 0, 0); *d = -X; }
}

// The planes as a ray sees them, in the game's order: origin and normal. The rotator-built normals keep their residuals
RLMAX_HD void ray_plane_at(int i, V3* o, V3* n) {
    const float X = c::ARENA_X_UU * c::UU_TO_BT, Z = c::ARENA_Z_UU * c::UU_TO_BT;
    constexpr float WALL_Z = 0x1.3d70a4p+4f;
    if (i == 0) { *o = v3(-X, 0.0f, WALL_Z); *n = v3(1.0f, 0.0f, 0.0f); }
    else if (i == 1) { *o = v3s(0.0f); *n = v3(0.0f, 0.0f, 1.0f); }
    else if (i == 2) { *o = v3(0.0f, 0.0f, Z); *n = v3(-0x1.135bd8p-47f, 0x1.777a5cp-24f, -1.0f); }
    else { *o = v3(X, 0.0f, WALL_Z); *n = v3(-0x1.fffffep-1f, 0.0f, 0x1.8p-23f); }
}

// The wheel-ray minimum hit distance squared, per car: ((hitbox_z * 0.5 - 2.25 uu) * UU_TO_BT)^2 with hitbox_z the full
// height with its skin, i.e. hitbox_half.z + margin - 2.25 uu in BT
// (the Octane gives the former constant 0.116684)
RLMAX_HD float wheel_ray_min_dist_sq(const CarGeo& g) {
    const float d = g.hitbox_half.z + g.margin - 2.25f * c::UU_TO_BT;
    return d * d;
}

struct WheelHit {
    bool ok;
    V3 point, normal;
    int body;  // -1 static world, 0 ball, 1.. car index + 1
};

// Closest accepted hit along from->to over the world, other cars and the ball. Hits closer to `from` than the
// minimum hit distance are rejected without hiding farther hits.
template <int NC>
RLMAX_HD WheelHit wheel_ray(const ArenaData& ad, V3 from, V3 to, int self_car, const Body* bodies, const CarGeo* geos,
                            float ball_radius, int cell) {
    WheelHit out{false, v3s(0), v3s(0), -2};
    V3 d = to - from;
    float l2 = len2(d);
    float min_frac2 = quirk::WHEEL_RAY_MIN_HIT_DIST ? wheel_ray_min_dist_sq(geos[self_car]) / l2 : 0.0f;
    float best = 1.0f;
    V3 best_n = v3s(0);
    int best_body = -2;
    auto take = [&](RayHit& h, int body) {
        if (h.ok && h.frac * h.frac >= min_frac2 && h.frac < best) {
            best = h.frac;
            best_n = h.n;
            best_body = body;
        }
    };
    for (int p = 0; p < NUM_PLANES; p++) {
        V3 po, pn;
        ray_plane_at(p, &po, &pn);
        RayHit h{1.0f, v3s(0), false};
        ray_plane_game(from, to, po, pn, &h);
        take(h, -1);
    }
    uint32_t b0 = ad.cell_off[cell], b1 = ad.cell_off[cell + 1];
    for (uint32_t k = b0; k < b1; k++) {
        RayHit h{1.0f, v3s(0), false};
        Tri tri = ad.tris[ad.cell_idx[k]];
        V3 org = v3s(0.0f);
        if (tri.flags & TF_ORIGIN_GOAL) {  // in its body's frame, as wheel_rays4
            V3 lv[3];
            goal_tri_local(tri, ad.goal_ly, lv);
            org = tri_origin(tri);
            tri.p0 = lv[0]; tri.p1 = lv[1]; tri.p2 = lv[2];
        }
        ray_tri(from - org, to - org, tri, &h);
        take(h, -1);
    }
    for (int ci = 0; ci < NC; ci++) {
        if (ci == self_car || !bodies[1 + ci].contact_response) continue;
        RayHit h{1.0f, v3s(0), false};
        if (ray_car_box_overlap(from, to, bodies[1 + ci], geos[ci]))
            ray_obb(from, to, car_obb(bodies[1 + ci], geos[ci]), &h);
        take(h, 1 + ci);
    }
    {
        RayHit h{1.0f, v3s(0), false};
        ray_sphere(from, to, bodies[0].pos, ball_radius, &h);
        take(h, 0);
    }
    if (best_body != -2) {
        out.ok = true;
        const float s_ = 1.0f - best;
        out.point = v3(from.x * s_ + to.x * best, from.y * s_ + to.y * best, from.z * s_ + to.z * best);
        out.normal = best_n;
        out.body = best_body;
    }
    return out;
}

#ifdef RLMAX_CHECK_RAYREJECT
// debug only (compile with -DRLMAX_CHECK_RAYREJECT): every ray the reach reject skips is cast anyway and must miss
struct RayRejectCount {
    long long n = 0, bad = 0;
    ~RayRejectCount() { fprintf(stderr, "rayreject: %lld rays compared, %lld hit\n", n, bad); }
};
inline RayRejectCount g_rayreject;
inline void rayreject_check(bool ok) { g_rayreject.n++; if (!ok) g_rayreject.bad++; }
#endif

// The four wheel rays of one car in one pass over the cell's triangles (each triangle loaded once for all four). Every
// ray sees the same tests in the same order as wheel_ray, so each result is identical to four wheel_ray calls.
template <int NC>
RLMAX_HD void wheel_rays4(const ArenaData& ad, const V3 from[4], const V3 to[4], int self_car, const Body* bodies,
                          const CarGeo* geos, float ball_radius, const float (*box)[6], int cell, WheelHit out[4]) {
    float min_frac2[4], best[4];
    V3 best_n[4];
    // fmin_/fmax_ can hide another ray's NaN, so use lengths to reject all non-finite endpoints.
    bool finite_rays = true;
    int best_body[4];
    for (int r = 0; r < 4; r++) {
        V3 d = to[r] - from[r];
        float l2 = len2(d);
        finite_rays &= l2 < FLT_BIG;
        min_frac2[r] = quirk::WHEEL_RAY_MIN_HIT_DIST ? wheel_ray_min_dist_sq(geos[self_car]) / l2 : 0.0f;
        best[r] = 1.0f;
        best_n[r] = v3s(0);
        best_body[r] = -2;
    }
    auto take = [&](int r, RayHit& h, int body) {
        if (h.ok && h.frac * h.frac >= min_frac2[r] && h.frac < best[r]) {
            best[r] = h.frac;
            best_n[r] = h.n;
            best_body[r] = body;
        }
    };
    // The game's ray walks the meshes in its own order and keeps the first of equal fractions; this list goes by mesh, so an equal
    // fraction on a mesh the game visits earlier takes over (the mirror seam at x = 0).
    uint32_t best_rank[4] = {0, 0, 0, 0};
    uint32_t b0 = ad.cell_off[cell], b1 = ad.cell_off[cell + 1];
    for (uint32_t k = b0; k < b1; k++) {
        Tri tri = ad.tris[ad.cell_idx[k]];
        // a goal triangle is cast in its body's frame, as the game's: the ray
        // less the origin against the body's own vertices, whose float rounding differs from the world frame's
        V3 org = v3s(0.0f);
        if (tri.flags & TF_ORIGIN_GOAL) {
            V3 lv[3];
            goal_tri_local(tri, ad.goal_ly, lv);
            org = tri_origin(tri);
            tri.p0 = lv[0]; tri.p1 = lv[1]; tri.p2 = lv[2];
        }
        const V3 nraw = tri_raw_normal(tri);
        for (int r = 0; r < 4; r++) {
            RayHit h{1.0f, v3s(0), false};
            ray_tri(from[r] - org, to[r] - org, tri, nraw, &h);
            if (!h.ok || !(h.frac * h.frac >= min_frac2[r])) continue;
            const uint32_t rank = tri_ray_rank(tri);
            if (!(h.frac < best[r] || (h.frac == best[r] && rank < best_rank[r]))) continue;
            if (!ray_leaf_box(from[r] - org, to[r] - org, ad.tri_box + 6u * ad.cell_idx[k])) continue;
            best[r] = h.frac;
            best_n[r] = h.n;
            best_body[r] = -1;
            best_rank[r] = rank;
        }
    }
    V3 rmn = from[0], rmx = from[0];
    for (int r = 0; r < 4; r++) {
        rmn = v3(fmin_(rmn.x, fmin_(from[r].x, to[r].x)), fmin_(rmn.y, fmin_(from[r].y, to[r].y)), fmin_(rmn.z, fmin_(from[r].z, to[r].z)));
        rmx = v3(fmax_(rmx.x, fmax_(from[r].x, to[r].x)), fmax_(rmx.y, fmax_(from[r].y, to[r].y)), fmax_(rmx.z, fmax_(from[r].z, to[r].z)));
    }
    // Each endpoint is at least 1/64 of an arena extent from the side walls and roof. With |y| < 2X
    // and |z| < X, their normal-rotation residuals and float error are under 0.001 BT; the margins
    // exceed 0.6 BT. The non-floor planes cannot intersect any of the four segments in this box.
    const float arena_x = c::ARENA_X_UU * c::UU_TO_BT;
    const float arena_z = c::ARENA_Z_UU * c::UU_TO_BT;
    const float edge_x = arena_x - arena_x * (1.0f / 64.0f);
    const float edge_z = arena_z - arena_z * (1.0f / 64.0f);
    const bool floor_only = finite_rays && rmn.x > -edge_x && rmx.x < edge_x && rmn.y > -2.0f * arena_x &&
                            rmx.y < 2.0f * arena_x && rmn.z > -arena_x && rmx.z < edge_z;
    if (floor_only) {
        for (int r = 0; r < 4; r++) {
            RayHit h{1.0f, v3s(0), false};
            ray_plane_game(from[r], to[r], v3s(0.0f), v3(0.0f, 0.0f, 1.0f), &h);
            take(r, h, -1);
        }
    } else {
        for (int r = 0; r < 4; r++)
            for (int p = 0; p < NUM_PLANES; p++) {
                V3 po, pn;
                ray_plane_at(p, &po, &pn);
                RayHit h{1.0f, v3s(0), false};
                ray_plane_game(from[r], to[r], po, pn, &h);
                take(r, h, -1);
            }
    }
    // Reach reject, once per (car, body) instead of once per ray. All eight ray points lie within `reach` of this car's origin, so the segments do too. ray_obb
    // and ray_sphere set `best` only at a marching point (from, or from + r*lambda with lambda in [0,1]) whose distance
    // to the rounded box / sphere is <= 0.001, i.e. within |hitbox_half| + margin + 0.001 of the box centre (resp. radius +
    // 0.001 of the ball centre); past reach + that + REACH_SLACK no such point exists, every exit leaves `best` untouched,
    // and take() ignores a !ok hit, so skipping the call changes nothing. REACH_SLACK (0.01 BT) is >> float error here.
    // The game's ray test walks the broadphase and reaches only the bodies whose stored box overlaps the four rays' box.
    // The box is the last sweep's, one tick behind a body a hard hit has just moved.
    auto box_hit = [&](int body) {
        if (!box) return true;
        const float* a = box[body];
        return !(a[3] < rmn.x || rmx.x < a[0] || a[4] < rmn.y || rmx.y < a[1] || a[5] < rmn.z || rmx.z < a[2]);
    };
    uint32_t near_cars = 0;
    bool near_ball = box_hit(0);
    for (int ci = 0; ci < NC; ci++)
        if (ci != self_car && bodies[1 + ci].contact_response && box_hit(1 + ci)) near_cars |= 1u << ci;
    if (near_cars || near_ball) {
        constexpr float REACH_SLACK = 0.01f;
        const V3 origin = bodies[1 + self_car].pos;
        float reach2 = 0.0f;
        for (int r = 0; r < 4; r++) {
            reach2 = fmax_(reach2, fmax_(len2(from[r] - origin), len2(to[r] - origin)));
        }
        const float reach = sqrtf(reach2) + REACH_SLACK;
        for (int ci = 0; ci < NC; ci++) {
            const uint32_t bit = 1u << ci;
            if (!(near_cars & bit)) continue;
            V3 obb_c = bodies[1 + ci].pos + bodies[1 + ci].r * geos[ci].hitbox_off;
            const float lim = reach + geos[ci].hitbox_len + geos[ci].margin;
            if (!(len2(obb_c - origin) <= lim * lim)) near_cars &= ~bit;
        }
        if (near_ball) {
            const float blim = reach + ball_radius;
            near_ball = len2(bodies[0].pos - origin) <= blim * blim;
        }
    }
    for (int r = 0; r < 4; r++) {
#ifndef RLMAX_CHECK_RAYREJECT
        if (near_cars)  // an empty conservative mask excludes every car for all four rays
#endif
        for (int ci = 0; ci < NC; ci++) {
#ifdef RLMAX_CHECK_RAYREJECT
            if (ci == self_car || !bodies[1 + ci].contact_response) continue;
            if (!(near_cars & (1u << ci))) {
                RayHit h{1.0f, v3s(0), false};
                ray_obb(from[r], to[r], car_obb(bodies[1 + ci], geos[ci]), &h);
                rayreject_check(!h.ok);
                continue;
            }
#else
            if (!(near_cars & (1u << ci))) continue;
#endif
            RayHit h{1.0f, v3s(0), false};
            if (ray_car_box_overlap(from[r], to[r], bodies[1 + ci], geos[ci]))
                ray_obb(from[r], to[r], car_obb(bodies[1 + ci], geos[ci]), &h);
            take(r, h, 1 + ci);
        }
        RayHit h{1.0f, v3s(0), false};
#ifdef RLMAX_CHECK_RAYREJECT
        if (!near_ball) {
            ray_sphere(from[r], to[r], bodies[0].pos, ball_radius, &h);
            rayreject_check(!h.ok);
            h = RayHit{1.0f, v3s(0), false};
        }
#endif
        if (near_ball) ray_sphere(from[r], to[r], bodies[0].pos, ball_radius, &h);
        take(r, h, 0);
        out[r] = WheelHit{false, v3s(0), v3s(0), -2};
        if (best_body[r] != -2) {
            out[r].ok = true;
            const float t_ = best[r], s_ = 1.0f - t_;
            out[r].point = v3(from[r].x * s_ + to[r].x * t_, from[r].y * s_ + to[r].y * t_, from[r].z * s_ + to[r].z * t_);
            out[r].normal = best_n[r];
            out[r].body = best_body[r];
        }
    }
}

// ---- per-wheel working state ---------------------------------------------------------------------------------

struct Wheel {
    bool hit;
    bool in_contact_world;
    int ground_body;
    V3 hard_point, axle_dir, contact_point, contact_normal, vel_at_contact, impulse;
    float susp_len, clipped_inv, susp_rel_vel, extra_pushback;
    float engine_force, brake, steer_angle, lat_friction, long_friction;
};

// resolve_single_collision
RLMAX_HD float impulse_denominator(const Body& b, bool dynamic, V3 pos, V3 n) {
    if (!dynamic) return 0.0f;
    V3 r0 = pos - b.pos;
    V3 c0 = cross(r0, n);
    V3 vec = cross(b.inv_i * c0, r0);  // the column form
    return b.inv_mass + dot(n, vec);
}

RLMAX_HD float resolve_single_collision(const Body& b1, const Body* b2, V3 pos, V3 n, float distance) {
    V3 rp1 = pos - b1.pos;
    V3 v1 = vel_at(b1, rp1);
    V3 v2 = b2 ? vel_at(*b2, pos - b2->pos) : v3s(0.0f);
    float rel = dot(n, v1 - v2);
    float pos_err = 0.2f * -distance / (1.0f / 60.0f);
    float vel_err = -rel;
    float d0 = impulse_denominator(b1, true, pos, n);
    float d1 = b2 ? impulse_denominator(*b2, true, pos, n) : 0.0f;
    float jinv = 1.0f / (d0 + d1);
    return fmax_(pos_err * jinv + vel_err * jinv, 0.0f);
}

RLMAX_HD float resolve_bilateral_fake_ground(const Body& b, V3 pos, V3 n) {
    V3 rel = pos - b.pos;
    V3 v1 = vel_at(b, rel);
    V3 aj = mul_t(b.r, cross(rel, n));
    float jd = b.inv_mass + dot(b.inv_i_local * aj, aj);
    return -0.2f * dot(n, v1) * (1.0f / jd);
}

}  // namespace rlmax
