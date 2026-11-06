#pragma once
#include "physics/tick.h"
namespace rlmax {
// Four lanes process contiguous cell ranges; overflow reruns the clean loop.
constexpr int WORLD_LANE_HIT_CAP = 4;
struct WorldLaneHit { MPoint point; uint32_t mesh; };
template<int NC> RLMAX_HD void world_lane_prefix(int i, ArenaState<NC>& s, const ArenaData& ad, const Body* bodies, const CarGeo* geo, ContactPoint (*car_world)[CAR_WORLD_SLOTS], int* car_world_n, bool* car_world_hit, V3* car_world_normal) {
        CarState& cs = s.cars[i];
        // car vs static: one manifold per (car, mesh component or plane) pair; the car is a compound, so each pair
        // refreshes, adds this tick's points, refreshes (manifold.h man_collide)
        car_world_hit[i] = false;
        car_world_normal[i] = v3(0, 0, 1);
        const Body& rb = bodies[1 + i];
        auto& w = cs.wman;
        if (!rb.contact_response || !quirk::CAR_WORLD_PERSISTENT) w.n = 0;
        if (rb.contact_response) {
            const float thr = geo[i].contact_threshold;
            V3 wpa[MANIFOLD_CAPACITY], wpb[MANIFOLD_CAPACITY];
            // a plane's manifold holds its points as offsets from the plane's origin
            auto refresh = [&](int q) {
                const bool pl = w.obj[q] >= 0x100u;
                const PlaneXf px = plane_xf(pl ? (int)(w.obj[q] - 0x100u) : 0);
                const V3 org = mesh_origin(w.obj[q]);
                man_refresh(w.m[q], &rb, nullptr, thr, wpa, wpb, pl ? &px : nullptr, w.obj[q] - 4u < 4u ? &org : nullptr);
            };
            if (quirk::COMPOUND_REFRESH_FIRST)
                for (int q = 0; q < CAR_WORLD_MANIFOLDS; q++)
                    if (q < (int)w.n && w.m[q].n) refresh(q);
            Obb ob = car_obb(rb, geo[i]);
            // The main floor and both goal-floor patches use the same reach test; the box cannot change between them.
            const bool floor_reach = obb_plane_reach(ob, v3(0.0f, 0.0f, 1.0f), 0.0f, thr);
            for (int p = 0; p < NUM_PLANES + 2; p++) {
                if (p == 0 || p >= NUM_PLANES) {
                    if (!floor_reach) continue;
                } else {
                    V3 pn; float pd;
                    arena_plane(p, &pn, &pd);
                    if (!obb_plane_reach(ob, pn, pd, thr)) continue;
                }
                // one new point per tick, the support vertex through the plane's own frame (contacts.h plane_point). Past
                // reach the refresh below already drops the plane's points (their distances are no less than the vertex's)
                MPoint np;
                const PlaneHit hit = plane_point(plane_xf(p), ob, rb.pos, thr, &np);
                if (hit == PLANE_FAR) {
                    for (int q = 0; q < (int)w.n; q++)
                        if (w.obj[q] == 0x100u + (uint32_t)p) w.m[q].n = 0;  // the game drops them whole
                } else if (hit == PLANE_POINT && !(np.d > thr)) {
                    if (Manifold* m = wman_get(w, 0x100u + (uint32_t)p)) man_add_point(*m, np, thr);
                }
            }
        }
}
template<int NC> RLMAX_HD void world_lane_fallback(int i, ArenaState<NC>& s, const ArenaData& ad, const Body* bodies, const CarGeo* geo, ContactPoint (*car_world)[CAR_WORLD_SLOTS], int* car_world_n, bool* car_world_hit, V3* car_world_normal, EpaScratch& scratch) {
    const Body& rb=bodies[1+i]; auto& w=s.cars[i].wman;
    if (!rb.contact_response) return;
    const float thr=geo[i].contact_threshold; Obb ob=car_obb(rb,geo[i]);
            // the convex-concave callback takes only triangles that touch the box's bounding box (TestTriangleAgainstAabb2)
            V3 bmn, bmx;
            obb_aabb(ob, &bmn, &bmx);
            const V3 ext = obb_extent(ob);
            int cell = cell_of(ad, ob.c);
            for (uint32_t t = ad.cell_off[cell]; t < ad.cell_off[cell + 1]; t++) {
                const uint32_t tri_index = ad.cell_idx[t];
                const Tri& tri = ad.tris[tri_index];
                // Test goal triangles in the body's frame: subtract its origin from the box's y
                // before comparing against the local vertex y.
                if (quirk::TRI_AABB_GATE_CAR) {
                    // A validated leaf is a conservative reject for this triangle.
                    if ((tri.flags & (TF_ORIGIN_GOAL | TF_LEAF_CONTAINS)) == TF_LEAF_CONTAINS) {
                        const float* leaf = ad.tri_box + 6u * tri_index;
                        if (leaf[2] > bmx.z || leaf[5] < bmn.z || leaf[0] > bmx.x || leaf[3] < bmn.x || leaf[1] > bmx.y || leaf[4] < bmn.y) continue;
                    }
                    if (tri.flags & TF_ORIGIN_GOAL) {
                        V3 lv[3];
                        goal_tri_local(tri, ad.goal_ly, lv);
                        const float cy = ob.c.y - tri_origin(tri).y;
                        if (!tri_aabb_overlap(lv[0], lv[1], lv[2], v3(bmn.x, cy - ext.y, bmn.z), v3(bmx.x, cy + ext.y, bmx.z))) continue;
                    } else if (!tri_aabb_overlap(tri.p0, tri.p1, tri.p2, bmn, bmx)) {
                        continue;
                    }
                }
                Contact cc = obb_tri(ob, tri, ad.goal_ly, thr, scratch);
                if (!cc.ok || cc.d > thr) continue;
                // the manifold keeps the point on the box from before the edge adjustment, and a goal's point on the
                // world as an offset from the body's origin
                const V3 pa = cc.pb + cc.n * cc.d;
                adjust_internal_edge(tri, &cc, ad.goal_ly);
                const uint32_t mesh = tri_mesh(tri);
                if (Manifold* m = wman_get(w, mesh))
                    man_add_point(*m, MPoint{mul_t(rb.r, pa - rb.pos), (tri.flags & TF_ORIGIN_GOAL) ? cc.pb - tri_origin(tri) : cc.pb, cc.n, cc.d}, thr);
            }
}
// Only hit records are retained; all filtering and contact arithmetic below
// are the clean loop's operations. A full list signals exact serial fallback.
template<int NC> RLMAX_HD bool world_lane_scan(int i,const ArenaData& ad,const Body* bodies,const CarGeo* geo,
                                               uint32_t first,uint32_t last,WorldLaneHit* hits,int& count,EpaScratch& scratch) {
    count=0; const Body& rb=bodies[1+i]; if (!rb.contact_response) return false;
    const float thr=geo[i].contact_threshold; const Obb ob=car_obb(rb,geo[i]);
    V3 bmn,bmx; obb_aabb(ob,&bmn,&bmx); const V3 ext=obb_extent(ob);
    for (uint32_t t=first;t<last;++t) {
                const uint32_t tri_index = ad.cell_idx[t];
                const Tri& tri = ad.tris[tri_index];
                // Test goal triangles in the body's frame: subtract its origin from the box's y
                // before comparing against the local vertex y.
                if (quirk::TRI_AABB_GATE_CAR) {
                    // A validated leaf is a conservative reject for this triangle.
                    if ((tri.flags & (TF_ORIGIN_GOAL | TF_LEAF_CONTAINS)) == TF_LEAF_CONTAINS) {
                        const float* leaf = ad.tri_box + 6u * tri_index;
                        if (leaf[2] > bmx.z || leaf[5] < bmn.z || leaf[0] > bmx.x || leaf[3] < bmn.x || leaf[1] > bmx.y || leaf[4] < bmn.y) continue;
                    }
                    if (tri.flags & TF_ORIGIN_GOAL) {
                        V3 lv[3];
                        goal_tri_local(tri, ad.goal_ly, lv);
                        const float cy = ob.c.y - tri_origin(tri).y;
                        if (!tri_aabb_overlap(lv[0], lv[1], lv[2], v3(bmn.x, cy - ext.y, bmn.z), v3(bmx.x, cy + ext.y, bmx.z))) continue;
                    } else if (!tri_aabb_overlap(tri.p0, tri.p1, tri.p2, bmn, bmx)) {
                        continue;
                    }
                }
        Contact cc=obb_tri(ob,tri,ad.goal_ly,thr,scratch);
        if (!cc.ok || cc.d>thr) continue;
        if (count==WORLD_LANE_HIT_CAP) return true;
        const V3 pa=cc.pb+cc.n*cc.d;
        adjust_internal_edge(tri,&cc,ad.goal_ly);
        hits[count++] = WorldLaneHit{MPoint{mul_t(rb.r,pa-rb.pos),
            (tri.flags & TF_ORIGIN_GOAL) ? cc.pb-tri_origin(tri) : cc.pb,cc.n,cc.d},tri_mesh(tri)};
    }
    return false;
}
template<int NC> RLMAX_HD void world_lane_finish(int i, ArenaState<NC>& s, const ArenaData& ad, const Body* bodies, const CarGeo* geo, ContactPoint (*car_world)[CAR_WORLD_SLOTS], int* car_world_n, bool* car_world_hit, V3* car_world_normal) {
    const Body& rb=bodies[1+i]; auto& w=s.cars[i].wman;
    if (rb.contact_response) {
        const float thr=geo[i].contact_threshold;
        V3 wpa[MANIFOLD_CAPACITY],wpb[MANIFOLD_CAPACITY];
        auto refresh=[&](int q) {
            const bool pl=w.obj[q]>=0x100u;
            const PlaneXf px=plane_xf(pl ? (int)(w.obj[q]-0x100u) : 0);
            const V3 org=mesh_origin(w.obj[q]);
            man_refresh(w.m[q],&rb,nullptr,thr,wpa,wpb,pl ? &px : nullptr,w.obj[q]-4u<4u ? &org : nullptr);
        };
            for (int q = 0; q < CAR_WORLD_MANIFOLDS; q++)
                if (q < (int)w.n) refresh(q);
            wman_release_empty(w);
        }
        // the solver's points straight from the refreshed manifolds, slot q * capacity + j (no response: w.n is 0)
        float deepest = FLT_BIG;
        car_world_n[i] = 0;
        // slot order q-major; the occupied slots are a prefix of each manifold and of the manifolds, so the first empty
        // one ends the loop (the old form tested all 16 slots: 90% of car-ticks hold none)
#pragma unroll
        for (int q = 0; q < CAR_WORLD_MANIFOLDS; q++) {
            if (q >= (int)w.n) break;
#pragma unroll
            for (int pi = 0; pi < MANIFOLD_CAPACITY; pi++) {
                if (pi >= (int)w.m[q].n) break;
                const bool pl = w.obj[q] >= 0x100u;
                const PlaneXf px = plane_xf(pl ? (int)(w.obj[q] - 0x100u) : 0);
                const V3 org = mesh_origin(w.obj[q]);
                const Contact cw = man_point(w.m[q], nullptr, pi, pl ? &px : nullptr, w.obj[q] - 4u < 4u ? &org : nullptr);
                ContactPoint& cp = car_world[i][car_world_n[i]++];
                cp.ok = true;
                cp.a = 1 + i;
                cp.b = -1;
                cp.pb = cw.pb;
                cp.n = cw.n;
                cp.d = cw.d;
                cp.pa = man_point_a(rb, w.m[q], pi);  // the manifold's point on A (Bullet m_positionWorldOnA)
                cp.friction = c::CAR_WORLD_FRICTION;
                cp.restitution = c::CAR_WORLD_RESTITUTION;
                cp.special = false;
                if (cw.ok && cw.d < deepest) {
                    deepest = cw.d;
                    car_world_hit[i] = true;
                    car_world_normal[i] = cw.n;
                }
            }
        }
        // Sum contact normals in static-object order, then normalize once.
        if (w.n) {
            V3 sum = v3s(0.0f);
            if (w.n == 1) {
                for (int pi = 0; pi < (int)w.m[0].n; pi++) sum += w.m[0].p[pi].n;
            } else {
                uint32_t last = 0;
                for (uint32_t n = 0; n < w.n; n++) {
                    int best = -1;
                    for (int q = 0; q < (int)w.n; q++) {
                        if (n && w.obj[q] <= last) continue;
                        if (best < 0 || w.obj[q] < w.obj[best]) best = q;
                    }
                    if (best < 0) break;
                    last = w.obj[best];
                    for (int pi = 0; pi < (int)w.m[best].n; pi++) sum += w.m[best].p[pi].n;
                }
            }
            const float l2 = len2(sum);
            if (l2 >= 1e-8f) car_world_normal[i] = sum * (1.0f / sqrtf(l2));
        }
        if (!quirk::CAR_WORLD_PERSISTENT) w.n = 0;
}
}
