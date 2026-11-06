// Bullet's persistent manifold (btPersistentManifold; bin 2a0060, 2adad0), for one body pair, in fixed storage. Points
// are kept in Bullet's compact cache order (appended; removal moves the last point into the hole), which is the order
// the solver sees and the order the area rule indexes. The pair's algorithm adds this tick's points first (nearest
// within the breaking threshold replaces it, else append, else the area rule over the four), then refreshes: every
// point re-projected through both bodies' current transforms, then culled in reverse order past the threshold or the
// tangential drift. A = first body, B = second; normals on B (world) point towards A; B may be the static world.
#pragma once

#include "body.h"
#include "contacts.h"
#include "state.h"

namespace rlmax {

// world <-> body frames; the static world is the identity
RLMAX_HD V3 man_to_local(const Body* b, V3 p) { return b ? mul_t(b->r, p - b->pos) : p; }
RLMAX_HD V3 man_to_world(const Body* b, V3 l) { return b ? b->pos + b->r * l : l; }

// btPersistentManifold::sortCachedPoints: keep the deepest point (by the cached distances), replace the one whose
// removal leaves the largest area; first index of the maximum on ties.
RLMAX_HD int man_sort_cached(const Manifold& m, V3 la, float d, bool on_b) {
    int maxpen = MANIFOLD_CAPACITY;
    float maxd = d;
    for (int i = 0; i < MANIFOLD_CAPACITY; i++)
        if (m.p[i].d < maxd) {
            maxpen = i;
            maxd = m.p[i].d;
        }
    const V3 p0 = on_b ? m.p[0].lb : m.p[0].la, p1 = on_b ? m.p[1].lb : m.p[1].la, p2 = on_b ? m.p[2].lb : m.p[2].la,
             p3 = on_b ? m.p[3].lb : m.p[3].la;
    float r0 = maxpen == 0 ? 0.0f : len2(cross(la - p1, p3 - p2));
    float r1 = maxpen == 1 ? 0.0f : len2(cross(la - p0, p3 - p2));
    float r2 = maxpen == 2 ? 0.0f : len2(cross(la - p0, p3 - p1));
    float r3 = maxpen == 3 ? 0.0f : len2(cross(la - p0, p2 - p1));
    int bi = 0;
    float bm = r0;
    if (r1 > bm) { bi = 1; bm = r1; }
    if (r2 > bm) { bi = 2; bm = r2; }
    if (r3 > bm) { bi = 3; }
    return bi;
}

RLMAX_HD void man_add_point(Manifold& m, const MPoint& np, float thr, bool on_b = false);

// btManifoldResult::addContactPoint into the pair's manifold: c.pb on B (world), c.n normal on B, c.d distance.
RLMAX_HD void man_add(Manifold& m, const Body* a, const Body* b, const Contact& c, float thr, bool on_b = false, const V3* pa_exact = nullptr) {
    if (!c.ok || c.d > thr) return;
    const V3 pa = pa_exact ? *pa_exact : c.pb + c.n * c.d;
    man_add_point(m, MPoint{man_to_local(a, pa), man_to_local(b, c.pb), c.n, c.d}, thr, on_b);
}

// The cache step of addContactPoint for a point already in the manifold's frames (the plane contacts build theirs).
RLMAX_HD void man_add_point(Manifold& m, const MPoint& np, float thr, bool on_b) {
    // getCacheEntry: the nearest cached point (A-local) within the threshold
    int hit = -1;
    float best = thr * thr;
    for (int i = 0; i < MANIFOLD_CAPACITY; i++) {
        if (i >= (int)m.n) break;
        float d2 = len2((on_b ? m.p[i].lb - np.lb : m.p[i].la - np.la));
        if (d2 < best) {
            best = d2;
            hit = i;
        }
    }
    if (hit < 0) hit = m.n < (uint32_t)MANIFOLD_CAPACITY ? (int)m.n++ : man_sort_cached(m, on_b ? np.lb : np.la, np.d, on_b);
    for (int i = 0; i < MANIFOLD_CAPACITY; i++)
        if (i == hit) m.p[i] = np;
}

// btPersistentManifold::refreshContactPoints
RLMAX_HD void man_refresh(Manifold& m, const Body* a, const Body* b, float thr, V3 pa_out[MANIFOLD_CAPACITY],
                          V3 pb_out[MANIFOLD_CAPACITY], const PlaneXf* px = nullptr, const V3* org = nullptr) {
    for (int i = 0; i < MANIFOLD_CAPACITY; i++) {
        if (i >= (int)m.n) break;
        pa_out[i] = man_to_world(a, m.p[i].la);
        pb_out[i] = px ? plane_to_world(*px, m.p[i].lb) : (org ? m.p[i].lb + *org : man_to_world(b, m.p[i].lb));
        m.p[i].d = dot(pa_out[i] - pb_out[i], m.p[i].n);
    }
    for (int i = MANIFOLD_CAPACITY - 1; i >= 0; i--) {
        if (i >= (int)m.n) continue;
        V3 proj = pa_out[i] - m.p[i].n * m.p[i].d;
        V3 diff = pb_out[i] - proj;
        bool cull = m.p[i].d > thr || dot(diff, diff) > thr * thr;
        if (!cull) continue;
        const int last = (int)m.n - 1;
        for (int j = 0; j < MANIFOLD_CAPACITY; j++)
            if (j == i) {
                m.p[j] = m.p[last];
                pa_out[j] = pa_out[last];
                pb_out[j] = pb_out[last];
            }
        m.n--;
    }
}

// One tick of a pair's algorithm on its manifold: this tick's points added, then refreshContactPoints. A compound
// body (the car: a box child at the hitbox offset) runs btCompoundCollisionAlgorithm, which refreshes every child
// manifold that holds points before the child algorithm runs, so its pairs refresh, add, refresh; a pair of plain
// shapes (the ball against a mesh or plane) adds, then refreshes.
template <int K>
RLMAX_HD void man_collide(Manifold& m, const Body* a, const Body* b, const Contact (&fresh)[K], float thr, bool compound,
                          V3 pa_out[MANIFOLD_CAPACITY], V3 pb_out[MANIFOLD_CAPACITY], bool on_b = false, const V3* pa_exact = nullptr) {  // pa_exact[q]: fresh[q]'s point on A as the box test computed it
    if (compound && m.n) man_refresh(m, a, b, thr, pa_out, pb_out);
    for (int q = 0; q < K; q++) man_add(m, a, b, fresh[q], thr, on_b, pa_exact ? &pa_exact[q] : nullptr);
    man_refresh(m, a, b, thr, pa_out, pb_out);
}

// Point i of a refreshed manifold as a solver contact (point on B, normal on B, distance).
RLMAX_HD Contact man_point(const Manifold& m, const Body* b, int i, const PlaneXf* px = nullptr, const V3* org = nullptr) {
    Contact c = no_contact();
    c.pb = px ? plane_to_world(*px, m.p[i].lb) : (org ? m.p[i].lb + *org : man_to_world(b, m.p[i].lb));
    c.n = m.p[i].n;
    c.d = m.p[i].d;
    c.ok = true;
    return c;
}

// Whether the body holds a point in its manifold with static object `obj`.
template <int M>
RLMAX_HD bool wman_holds(const WorldManifolds<M>& w, uint32_t obj) {
    for (int q = 0; q < M; q++)
        if (q < (int)w.n && w.obj[q] == obj) return w.m[q].n != 0;
    return false;
}

// The manifold for a body's pair with static object `obj`: the existing one, else a new one appended (the pair's
// algorithm creates it on first contact); null when all M are in use.
template <int M>
RLMAX_HD Manifold* wman_get(WorldManifolds<M>& w, uint32_t obj) {
    for (int q = 0; q < M; q++)
        if (q < (int)w.n && w.obj[q] == obj) return &w.m[q];
    if (w.n >= (uint32_t)M) return nullptr;
    const uint32_t q = w.n++;
    w.obj[q] = obj;
    w.m[q].n = 0;
    return &w.m[q];
}

// Release the empty manifolds, in reverse order, the last moving into the hole.
template <int M>
RLMAX_HD void wman_release_empty(WorldManifolds<M>& w) {
    for (int q = M - 1; q >= 0; q--) {
        if (q >= (int)w.n || w.m[q].n) continue;
        const uint32_t last = w.n - 1;
        w.m[q] = w.m[last];
        w.obj[q] = w.obj[last];
        w.n--;
    }
}

}  // namespace rlmax
