#pragma once
// Car-ball manifolds are per car; GPU staging uses independent lanes.
#include "../staged_tick.h"
namespace rlmax {
template <int NC>
RLMAX_HD void staged_car_ball(int i, ArenaState<NC>& s, const Body* bodies, const CarGeo* geo,
                              const TickConsts& k, CarSnap* snap, ContactPoint (*car_ball)[CAR_BALL_SLOTS],
                              int* car_ball_n, bool* car_ball_touch) {
    car_ball_touch[i] = false;
    const Body& bb = bodies[1 + i];
    const Body& ba = bodies[0];
    Manifold& m = s.ball_man[i];
    // the compound algorithm runs the box child only while its bounding box overlaps the ball's, and destroys the
    // child algorithm (releasing its manifold) when they stop overlapping
    const Obb cob = car_obb(bb, geo[i]);
    V3 cmn, cmx;
    obb_aabb(cob, &cmn, &cmx);
    const V3 rr = v3s(k.ball_radius + c::SPHERE_BOUND_PAD_BT);
    const bool aabbs = !quirk::CHILD_AABB_GATE_BALL || aabb_overlap(cmn, cmx, ba.pos - rr, ba.pos + rr);
    V3 wpa[MANIFOLD_CAPACITY], wpb[MANIFOLD_CAPACITY];
    if (!bb.contact_response || !aabbs) {
        if (m.n) m.n = 0;
    } else {
        const float thr = fmin_(k.ball_threshold, geo[i].contact_threshold);
        const float md_ = cob.margin + k.ball_radius + thr;
        const GjkSphereResult gs = gjk_box_sphere(cob.h, cob.margin, cob.r, cob.c, ba.pos, k.ball_radius, md_ * md_);
        if (gs.ok) {
            Contact cc[1] = {no_contact()};
            V3 on_ball = gs.on_ball;
            if (gs.contact && gs.d <= thr) {
                cc[0].ok = true;
                cc[0].n = -gs.normal;
                cc[0].d = gs.d;
                cc[0].pb = gs.on_ball + gs.normal * gs.d;
            }
            man_collide(m, &ba, &bb, cc, thr, quirk::COMPOUND_REFRESH_FIRST, wpa, wpb, true, &on_ball);
            car_ball_touch[i] = m.n > 0;
        } else {
            Contact cc[1] = {sphere_obb(ba.pos, k.ball_radius, cob, thr)};
            man_collide(m, &ba, &bb, cc, thr, quirk::COMPOUND_REFRESH_FIRST, wpa, wpb, true);
            car_ball_touch[i] = m.n > 0;
        }
    }
    if (car_ball_touch[i]) snap[i].q = bb.rot;
    // the solver's points straight from the refreshed manifold (cache order; m.n is 0 without a response)
    static_assert(CAR_BALL_SLOTS == MANIFOLD_CAPACITY, "one slot per manifold point");
    car_ball_n[i] = 0;
#pragma unroll
    for (int q = 0; q < CAR_BALL_SLOTS; q++) {
        if (q >= (int)m.n) break;
        const V3 pb = wpb[q], n = m.p[q].n;
        const float d = m.p[q].d;
        ContactPoint& cp = car_ball[i][car_ball_n[i]++];
        // the solver's first body is the car, so its normal points at the car; the manifold keeps the ball as A
        cp.ok = true;
        cp.a = 1 + i;
        cp.b = 0;
        cp.pa = pb;
        cp.pb = wpa[q];  // the manifold's point on the ball (Bullet m_positionWorldOnA), not the car's plus the normal depth
        cp.n = -n;
        cp.d = d;
        cp.friction = c::CAR_BALL_FRICTION;
        cp.restitution = c::CAR_BALL_RESTITUTION;
        cp.special = false;
    }
}
}
