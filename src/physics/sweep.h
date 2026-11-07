// RLMAX broadphase order (P4 lead 16): the order the solver takes its contact rows in, which in Bullet 2.82 is the order
// of the dispatcher's manifold array, sorted by simulation island. Both come from the broadphase: btAxisSweep3 keeps
// 16-bit quantised AABB edges sorted per axis by insertion sort, adds and removes overlapping pairs as edges pass each
// other, and its pair cache removes by swap-with-last; a manifold is made when its pair is first dispatched and swap-
// removed when the pair goes (or a compound child's AABB leaves the other shape). This code ports exactly that; it is written over RLMAX's state in that order. Only the order is
// modelled: the contact points stay where manifold.h and tick.h compute them.
//
// Handles (1-based): 16 soccar components, 4 planes (floor, ceiling, -x wall, +x wall), the ball,
// then the cars. The statics never move; their edges are constants (sweep_tables.h). The state per arena is each
// dynamic edge's place in its axis and quantised position (the next tick's sort starts from them), the pair array and
// the manifold array, as byte keys through six cars and 16-bit keys for eight cars.
#pragma once

#include "car.h"
#include "state.h"
#include "sweep_tables.h"

namespace rlmax {

constexpr int SW_STATICS = 22;
constexpr int SW_BALL = SW_STATICS + 1;  // the ball's handle; car i is SW_BALL + 1 + i

// The world box is +-HALF about the origin, positions 16 bits, the low bit is the max-edge flag.
constexpr float SW_HALF = 10485.740234375f;

// A pair of handles in one byte: (high - ball) * 29 + low, the high handle a dynamic body (two statics never pair)
constexpr int SW_KEY_STRIDE = 29;
RLMAX_HD uint8_t sw_key(int a, int b) {
    const int lo = a < b ? a : b, hi = a < b ? b : a;
    return (uint8_t)((hi - SW_BALL) * SW_KEY_STRIDE + lo);
}
RLMAX_HD int sw_lo(int k) { return k % SW_KEY_STRIDE; }
RLMAX_HD int sw_hi(int k) { return SW_BALL + k / SW_KEY_STRIDE; }

// Seven and eight cars need handles through 30 and 31 (the stride-29 byte key reaches only 28): stride 31. Seven cars'
// largest key is 6 * 31 + 29 = 215, so their key stays a byte; eight cars' keys are 16 bits (SweepKey).
template <int NC> RLMAX_HD SweepKey<NC> sw_key_for(int a, int b) {
    if constexpr (NC >= 7) {
        const int lo = a < b ? a : b, hi = a < b ? b : a;
        return (SweepKey<NC>)((hi - SW_BALL) * 31 + lo);
    } else return sw_key(a, b);
}
template <int NC> RLMAX_HD int sw_lo_for(SweepKey<NC> k) {
    if constexpr (NC >= 7) return k % 31;
    else return sw_lo(k);
}
template <int NC> RLMAX_HD int sw_hi_for(SweepKey<NC> k) {
    if constexpr (NC >= 7) return SW_BALL + k / 31;
    else return sw_hi(k);
}

RLMAX_HD uint16_t sw_quant(float p, uint32_t is_max) {
    const float q = 65535.0f / (SW_HALF + SW_HALF);
    float scaled = (p - (-SW_HALF)) * q;
    scaled = scaled >= 0.0f ? fmin_(scaled, 65535.0f) : 0.0f;
    return (uint16_t)(((uint32_t)(int)scaled & 0xfffeu) | is_max);
}

// What the sweep needs of one tick's bodies: each dynamic body's proxy AABB quantised (the union of the box now and the
// box one tick on, plus the contact breaking threshold), the cars' child boxes and the ball's box for the dispatch
// gates, and who takes part.
template <int NC>
struct SweepIn {
    uint32_t qedge[1 + NC][3];  // quantised minimum | maximum << 16
    V3 child_min[NC], child_max[NC];
    V3 ball_min, ball_max;
    float ball_e;                 // the ball's bound half extent
    const Body* bodies;           // the child boxes' centres and axes are rebuilt from these where a manifold is made
    const CarGeo* geo;
    bool active[1 + NC], response[1 + NC];
    uint32_t predictive;          // bodies with a predictive world contact this tick
    uint32_t visit;               // the cars' visit order (ArenaState::visit): the order of the sweep's object list
    uint32_t ball_planes_held;    // bit p: the ball holds a manifold with arena plane p (handle 17 + p), see ball_held_planes
};

// A static handle's raw box against a car's child box (inclusive).
RLMAX_HD bool sw_static_gate(int h, V3 cmn, V3 cmx) {
    if (h <= 16) {
        const float* t = SW_MESH_AABB[h - 1];
        return aabb_overlap(cmn, cmx, v3(t[0], t[1], t[2]), v3(t[3], t[4], t[5]));
    }
    // a plane: its getAabb is a zero-thickness finite rectangle (binary 2a70b0), taken in the car's
    // frame by the compound's tree query, where any tilt of the car gives it the thickness of the rectangle times the tilt; the
    // exact test (the rectangle's box on the car's axes, 36 of 18,710 wisp pairs differ from the game) measures the same, and
    // a car-plane manifold as long as the pair lives (static_id >= 100)
    return true;
}

// Before creating a mesh child algorithm, Bullet queries the compound's tree
// with the mesh AABB in the car's local frame. The later world AABB test alone
// can admit the pair one tick too early and change dispatcher manifold order.
template <int NC>
RLMAX_HD bool sw_static_local_gate(const SweepIn<NC>& in, int h, int b) {
    if (h > 16) return true;
    const float* t = SW_MESH_AABB[h - 1];
    const V3 c = v3((t[0] + t[3]) * 0.5f, (t[1] + t[4]) * 0.5f, (t[2] + t[5]) * 0.5f);
    const V3 e = v3((t[3] - t[0]) * 0.5f, (t[4] - t[1]) * 0.5f, (t[5] - t[2]) * 0.5f);
    const Body& body = in.bodies[b];
    const CarGeo& geo = in.geo[b - 1];
    const V3 lc = mul_t(body.r, c) + mul_t(body.r, -body.pos);
    const V3 half = geo.hitbox_half + v3s(geo.margin);
    for (int k = 0; k < 3; k++) {
        const V3 a = k == 0 ? body.r.c0 : (k == 1 ? body.r.c1 : body.r.c2);
        const float r = absf(a.x) * e.x + absf(a.y) * e.y + absf(a.z) * e.z;
        if (vget(lc, k) - r > vget(geo.hitbox_off, k) + vget(half, k) ||
            vget(lc, k) + r < vget(geo.hitbox_off, k) - vget(half, k)) return false;
    }
    return true;
}

// A compound child algorithm is made only for a child the compound's tree query returns, in the car's own frame: the other
// shape's box there against the child's box. Tested on the car's three axes: the ball as a sphere of its bound radius, the
// other car as its box projected on them. a is the car, b the other body (0: the ball, else a car, 1-based).
template <int NC>
RLMAX_HD bool sw_local_overlap(const SweepIn<NC>& in, int a, int b) {
    const Obb oa = car_obb(in.bodies[a], in.geo[a - 1]);
    const V3 d = (b == 0 ? in.bodies[0].pos : car_obb(in.bodies[b], in.geo[b - 1]).c) - oa.c;
    const V3 ha = obb_full_half(oa);
    for (int k = 0; k < 3; k++) {
        const V3 ak = k == 0 ? oa.r.c0 : (k == 1 ? oa.r.c1 : oa.r.c2);
        float e;
        if (b == 0) {
            e = in.ball_e;
        } else {
            const Obb ob = car_obb(in.bodies[b], in.geo[b - 1]);
            const V3 hb = obb_full_half(ob);
            e = absf(dot(ak, ob.r.c0)) * hb.x + absf(dot(ak, ob.r.c1)) * hb.y + absf(dot(ak, ob.r.c2)) * hb.z;
        }
        if (absf(dot(d, ak)) > vget(ha, k) + e) return false;
    }
    return true;
}

// Each static handle's min and max edge rank among the statics' edges, per axis (the inverse of SW_STATIC_EDGE_H)
struct SwRank {
    uint8_t k[SW_STATICS + 1][3][2];
};
constexpr SwRank sw_make_rank() {
    SwRank r{};
    for (int axis = 0; axis < 3; axis++)
        for (int k = 0; k < 2 * SW_STATICS; k++)
            r.k[SW_STATIC_EDGE_H[axis][k]][axis][SW_STATIC_EDGE_POS[axis][k] & 1] = (uint8_t)k;
    return r;
}
RLMAX_CONST SwRank SW_RANK = sw_make_rank();

RLMAX_HD int sw_popcount(uint64_t bits) {
#if defined(__CUDA_ARCH__)
    return __popcll(bits);
#elif defined(__clang__) || defined(__GNUC__)
    return __builtin_popcountll(bits);
#else
    int count = 0;
    while (bits) { bits &= bits - 1; ++count; }
    return count;
#endif
}

template <int NC, bool SlotOnly, bool Lazy = (NC > 2 && !SlotOnly)>
struct SweepStorage {};
template <int NC, bool Lazy>
struct SweepStorage<NC, true, Lazy> { SweepState<NC>* saved; uint64_t slot_mask[3]; };
template <int NC>
struct SweepStorage<NC, false, true> {
    SweepState<NC>* saved;
    uint64_t occupied[3];
    unsigned loaded;
};

template <int CAP, bool Small = (CAP <= 64)>
struct SweepActive {
    bool values[CAP];
    RLMAX_HD void clear() {}
    RLMAX_HD bool get(int i) const { return values[i]; }
    RLMAX_HD void set(int i, bool value) { values[i] = value; }
};
template <int CAP>
struct SweepActive<CAP, true> {
    uint64_t bits;
    RLMAX_HD void clear() { bits = 0; }
    RLMAX_HD bool get(int i) const { return (bits >> i) & 1u; }
    RLMAX_HD void set(int i, bool value) {
        const uint64_t bit = uint64_t(1) << i;
        bits = value ? bits | bit : bits & ~bit;
    }
};

template <int NC, bool SlotOnly = false>
struct Sweep : SweepStorage<NC, SlotOnly> {
    static constexpr int NB = 1 + NC;
    static constexpr int ND = 2 * NB;           // dynamic edges per axis: 2 * body + is_max
    static constexpr int NH = SW_STATICS + NB;  // live handles
    static constexpr int NE = 2 * NH + 2;       // edges per axis with the two sentinels
    static constexpr int CAP = sweep_man_cap<NC>();
    static constexpr int PAIR_CAP = sweep_cap<NC>();
    static_assert(NH <= (NC >= 7 ? 31 : SW_KEY_STRIDE), "pair key stride");

    // a fresh sweep's lists; a valid sweep's slot form keeps dynamic edge d's place in hnd[axis][d] and its position in
    // pos[axis][d] (the lists are not built there, so the frame holds one of the two)
    uint16_t pos[3][SlotOnly ? 1 : NE];
    uint8_t hnd[3][SlotOnly ? 1 : NE];
    uint8_t mn[NB][3], mx[NB][3];  // dynamic handles' edge indices per axis
    int nh;                                 // full lists: handle count; slots: ready-axis bits
    SweepKey<NC>* pair;  // persistent pair cache, in dispatcher order
    int np;
    SweepKey<NC>* man;   // persistent manifold order
    int nm;
    SweepActive<CAP> act;

    RLMAX_HD int find_pair(SweepKey<NC> k) const {
        for (int i = 0; i < np; i++)
            if (pair[i] == k) return i;
        return -1;
    }
    RLMAX_HD int find_man(SweepKey<NC> k) const {
        for (int i = 0; i < nm; i++)
            if (man[i] == k) return i;
        return -1;
    }
    // swap-with-last, as the dispatcher's releaseManifold (and the active flag goes with the entry)
    RLMAX_HD void release(SweepKey<NC> k, bool active_flags = true) {
        const int i = find_man(k);
        if (i < 0) return;
        man[i] = man[nm - 1];
        // Edge sorting runs before active flags are initialized; dispatch runs after.
        if (active_flags) act.set(i, act.get(nm - 1));
        nm--;
    }

    // OverlappingPairCache::add_overlapping_pair: two statics never pair; an existing pair stays where it is
    RLMAX_HD void add_pair(int a, int b) {
        if ((a > b ? a : b) <= SW_STATICS) return;
        const SweepKey<NC> k = sw_key_for<NC>(a, b);
        if (find_pair(k) >= 0 || np >= PAIR_CAP) return;
        pair[np++] = k;
    }
    // remove_overlapping_pair (swap-with-last) and the manifold's release
    RLMAX_HD void remove_pair(int a, int b) {
        const SweepKey<NC> k = sw_key_for<NC>(a, b);
        const int i = find_pair(k);
        if (i < 0) return;
        pair[i] = pair[np - 1];
        np--;
        release(k, false);
    }

    // Minimum edges are even and maximum edges are odd, so opposite types
    // cannot tie. On a completed axis their value order equals their slot order.
    RLMAX_HD uint16_t dynamic_edge_value(int h, int axis, bool is_max) const {
        if constexpr (SlotOnly) {
            const uint32_t edge = this->saved->edge[h - SW_BALL][axis];
            return (uint16_t)(is_max ? edge >> 16 : edge);
        }
        if constexpr (NC > 2 && !SlotOnly) {
            if (!(this->loaded & (1u << axis))) {
                const uint32_t edge = this->saved->edge[h - SW_BALL][axis];
                return (uint16_t)(is_max ? edge >> 16 : edge);
            }
        }
        return pos[axis][is_max ? mx[h - SW_BALL][axis] : mn[h - SW_BALL][axis]];
    }
    RLMAX_HD uint16_t edge_value(int h, int axis, bool is_max) const {
        if (h >= SW_BALL) return dynamic_edge_value(h, axis, is_max);
        return SW_STATIC_EDGE_POS[axis][SW_RANK.k[h][axis][is_max ? 1 : 0]];
    }
    // a is the moving handle and is always dynamic.
    RLMAX_HD bool overlap_2d(int a, int b, int axis) const {
        const int a1 = (1 << axis) & 3, a2 = (1 << a1) & 3;
        if (dynamic_edge_value(a, a1, true) < edge_value(b, a1, false) ||
            edge_value(b, a1, true) < dynamic_edge_value(a, a1, false)) return false;
        return !(dynamic_edge_value(a, a2, true) < edge_value(b, a2, false) ||
                 edge_value(b, a2, true) < dynamic_edge_value(a, a2, false));
    }
    RLMAX_HD void swap_edges(int axis, int i, int j) {
        const uint16_t p = pos[axis][i];
        const uint8_t h = hnd[axis][i];
        pos[axis][i] = pos[axis][j];
        hnd[axis][i] = hnd[axis][j];
        pos[axis][j] = p;
        hnd[axis][j] = h;
    }

    // sortMinDown / sortMaxDown: the moving edge walks down while it is below its neighbour; sortMaxUp / sortMinUp walk
    // up while it is not below it, stopping at the max sentinel (handle 0). A min edge passing down over a max edge
    // (or a max edge passing up over a min edge) adds the pair when the other two axes overlap; the reverse removes it.
    RLMAX_HD void sort_down(int axis, int index, bool moving_is_max, bool update) {
        const int moving = hnd[axis][index];
        while (index > 0 && pos[axis][index] < pos[axis][index - 1]) {
            const int prev = hnd[axis][index - 1];
            const bool prev_is_max = (pos[axis][index - 1] & 1) != 0;
            if (prev_is_max != moving_is_max) {
                if (update && overlap_2d(moving, prev, axis)) {
                    if (moving_is_max) remove_pair(moving, prev);
                    else add_pair(moving, prev);
                }
            }
            if (prev >= SW_BALL) {
                if (prev_is_max) mx[prev - SW_BALL][axis]++;
                else mn[prev - SW_BALL][axis]++;
            }
            if (moving_is_max) mx[moving - SW_BALL][axis]--;
            else mn[moving - SW_BALL][axis]--;
            swap_edges(axis, index, index - 1);
            index--;
        }
    }
    RLMAX_HD void sort_up(int axis, int index, bool moving_is_max, bool update) {
        const int moving = hnd[axis][index];
        const int len = 2 * nh + 2;
        while (index + 1 < len && hnd[axis][index + 1] != 0 && pos[axis][index] >= pos[axis][index + 1]) {
            const int next = hnd[axis][index + 1];
            const bool next_is_max = (pos[axis][index + 1] & 1) != 0;
            if (next_is_max != moving_is_max) {
                if (update && overlap_2d(moving, next, axis)) {
                    if (moving_is_max) add_pair(moving, next);
                    else remove_pair(moving, next);
                }
            }
            if (next >= SW_BALL) {
                if (next_is_max) mx[next - SW_BALL][axis]--;
                else mn[next - SW_BALL][axis]--;
            }
            if (moving_is_max) mx[moving - SW_BALL][axis]++;
            else mn[moving - SW_BALL][axis]++;
            swap_edges(axis, index, index + 1);
            index++;
        }
    }

    // addHandle: the two edges go just inside the max sentinel, axes 0 and 1 sort down without events, axis 2 with them
    RLMAX_HD void add_handle(int h, const uint32_t* qedge, bool events = true) {
        nh++;
        const int limit = 2 * nh;
        for (int axis = 0; axis < 3; axis++) {
            pos[axis][limit + 1] = pos[axis][limit - 1];
            hnd[axis][limit + 1] = hnd[axis][limit - 1];
            pos[axis][limit - 1] = (uint16_t)qedge[axis];
            hnd[axis][limit - 1] = (uint8_t)h;
            pos[axis][limit] = (uint16_t)(qedge[axis] >> 16);
            hnd[axis][limit] = (uint8_t)h;
            mn[h - SW_BALL][axis] = (uint8_t)(limit - 1);
            mx[h - SW_BALL][axis] = (uint8_t)limit;
        }
        for (int axis = 0; axis < 3; axis++) {
            sort_down(axis, mn[h - SW_BALL][axis], false, events && axis == 2);
            sort_down(axis, mx[h - SW_BALL][axis], true, events && axis == 2);
        }
    }

    // Dynamic slots form a bit set; all remaining slots are immutable static
    // edges. Build it once per axis, then an edge can test its nearest neighbour
    // without rebuilding the static list. Falling back preserves crossing order.
    RLMAX_HD bool crosses_neighbour(int body, int axis, int side, uint16_t value, uint32_t new_edge) const {
        const int slot = this->saved->slot[body][2 * axis + side];
        const uint32_t old_edge = this->saved->edge[body][axis];
        const uint16_t old = (uint16_t)(side ? old_edge >> 16 : old_edge);
        if (value == old) return false;
        const bool up = value > old;
        const int target = slot + (up ? 1 : -1);
        if (target <= 0 || target >= NE - 1) return false;
        const uint64_t bit = uint64_t(1) << target;
        const uint64_t mask = this->occupied[axis];
        uint16_t other = 0;
        if (mask & bit) {
            for (int b = 0; b < NB; ++b) {
                for (int e = 0; e < 2; ++e) {
                    if (this->saved->slot[b][2 * axis + e] != target) continue;
                    const uint32_t edge = b == body ? new_edge : this->saved->edge[b][axis];
                    other = (uint16_t)(e ? edge >> 16 : edge);
                    return up ? value >= other : value < other;
                }
            }
        } else {
            const uint64_t below = mask & (bit - 1);
            other = SW_STATIC_EDGE_POS[axis][target - 1 - sw_popcount(below)];
        }
        return up ? value >= other : value < other;
    }

    // setAabb: expand first (adds), then shrink (removes)
    RLMAX_HD void update_axis(int h, int axis, uint32_t edge) {
        const uint16_t qmin = (uint16_t)edge, qmax = (uint16_t)(edge >> 16);
        if constexpr (NC > 2 && !SlotOnly) if (!(this->loaded & (1u << axis))) {
            const int body = h - SW_BALL;
            if (this->saved->edge[body][axis] == edge) return;
            if (!(this->loaded & (8u << axis))) {
                uint64_t mask = 0;
                for (int b = 0; b < NB; ++b) {
                    mask |= uint64_t(1) << this->saved->slot[b][2 * axis];
                    mask |= uint64_t(1) << this->saved->slot[b][2 * axis + 1];
                }
                this->occupied[axis] = mask;
                this->loaded |= 8u << axis;
            }
            if (!crosses_neighbour(body, axis, 0, qmin, edge) &&
                !crosses_neighbour(body, axis, 1, qmax, edge)) {
                this->saved->edge[body][axis] = edge;
                return;
            }
            ensure_axis(axis);
        }
        const int emin = mn[h - SW_BALL][axis], emax = mx[h - SW_BALL][axis];
        const int dmin = (int)qmin - (int)pos[axis][emin];
        const int dmax = (int)qmax - (int)pos[axis][emax];
        if (dmin == 0 && dmax == 0) return;
        pos[axis][emin] = qmin;
        pos[axis][emax] = qmax;
        if (dmin < 0) sort_down(axis, emin, false, true);
        if (dmax > 0) sort_up(axis, emax, true, true);
        if (dmin > 0) sort_up(axis, emin, false, true);
        if (dmax < 0) sort_down(axis, emax, true, true);
    }

    RLMAX_HD void index_edges() {
        for (int axis = 0; axis < 3; axis++)
            for (int e = 1; e <= 2 * nh; e++) {
                const int h = hnd[axis][e];
                if (h < SW_BALL) continue;
                if (pos[axis][e] & 1) mx[h - SW_BALL][axis] = (uint8_t)e;
                else mn[h - SW_BALL][axis] = (uint8_t)e;
            }
    }

    // The sentinels and the statics' edges (a fresh sweep before any dynamic handle)
    RLMAX_HD void statics_only() {
        for (int axis = 0; axis < 3; axis++) {
            pos[axis][0] = 0;
            hnd[axis][0] = 0;
            for (int e = 0; e < 2 * SW_STATICS; e++) {
                pos[axis][1 + e] = SW_STATIC_EDGE_POS[axis][e];
                hnd[axis][1 + e] = SW_STATIC_EDGE_H[axis][e];
            }
            pos[axis][2 * SW_STATICS + 1] = 0xffff;
            hnd[axis][2 * SW_STATICS + 1] = 0;
        }
        nh = SW_STATICS;
    }

    // The lists as the last tick left them: dynamic edges at their stored places, the statics' edges (their relative
    // order never changes) in the places left
    RLMAX_HD void load_axis(const SweepState<NC>& st, int axis) {
        nh = NH;
        {
            uint64_t taken = 0;
            for (int b = 0; b < NB; b++) {
                const int h = SW_BALL + b;
                const int s0 = st.slot[b][axis * 2], s1 = st.slot[b][axis * 2 + 1];
                taken |= (uint64_t(1) << s0) | (uint64_t(1) << s1);
                pos[axis][s0] = (uint16_t)(st.edge[b][axis] & 0xffffu);
                pos[axis][s1] = (uint16_t)(st.edge[b][axis] >> 16);
                hnd[axis][s0] = hnd[axis][s1] = (uint8_t)h;
                mn[h - SW_BALL][axis] = (uint8_t)s0;
                mx[h - SW_BALL][axis] = (uint8_t)s1;
            }
            pos[axis][0] = 0;
            hnd[axis][0] = 0;
            pos[axis][NE - 1] = 0xffff;
            hnd[axis][NE - 1] = 0;
            int k = 0;
#if defined(__CUDACC__)
#pragma unroll 1
#endif
            for (int e = 1; e < NE - 1; e++) {
                if ((taken >> e) & 1u) continue;
                const uint16_t edge = SW_STATIC_EDGE_POS[axis][k];
                const uint8_t handle = SW_STATIC_EDGE_H[axis][k];
                pos[axis][e] = edge;
                hnd[axis][e] = handle;
                k++;
            }
        }
        this->loaded |= 1u << axis;
    }

    RLMAX_HD void ensure_axis(int axis) {
        if constexpr (NC > 2 && !SlotOnly) if (!(this->loaded & (1u << axis))) load_axis(*this->saved, axis);
    }

    RLMAX_HD void store(SweepState<NC>& st) const {
        for (int axis = 0; axis < 3; axis++) {
            if constexpr (NC > 2 && !SlotOnly) if (!(this->loaded & (1u << axis))) continue;
            for (int b = 0; b < NB; b++) {
                const int s0 = mn[b][axis], s1 = mx[b][axis];
                st.slot[b][axis * 2] = (uint8_t)s0;
                st.slot[b][axis * 2 + 1] = (uint8_t)s1;
                st.edge[b][axis] = (uint32_t)pos[axis][s0] | ((uint32_t)pos[axis][s1] << 16);
            }
        }
    }

    // The slot form of a valid sweep: the lists are never built. The statics' relative order never changes, so the edge at
    // a place is a dynamic edge (its stored place) or the static of rank (place - 1 - dynamic edges below it), and the sorts
    // below make the same comparisons, swaps and pair events as sort_down / sort_up on the lists (P3 §6: rebuilding the
    // lists each tick was 9.5% of the 1v1 tick's cycles). An edge is d >= 0 (dynamic) or -1 - k (the static of rank k).
    RLMAX_HD int slot_pos(int axis, int d) const {
        return this->saved->slot[d / 2][2 * axis + (d & 1)];
    }
    RLMAX_HD void set_slot_pos(int axis, int d, int pos) {
        this->saved->slot[d / 2][2 * axis + (d & 1)] = (uint8_t)pos;
    }
    RLMAX_HD uint16_t slot_value(int axis, int d) const {
        const uint32_t edge = this->saved->edge[d / 2][axis];
        return (uint16_t)((d & 1) ? edge >> 16 : edge);
    }
    // Masks are written on demand; no per-axis clearing is needed.
    RLMAX_HD void slots_load(const SweepState<NC>&) {
        this->nh = 0;
    }
    RLMAX_HD void prepare_slot_mask(int axis) {
        uint64_t mask = 0;
        for (int b = 0; b < NB; ++b) {
            mask |= uint64_t(1) << this->saved->slot[b][2 * axis];
            mask |= uint64_t(1) << this->saved->slot[b][2 * axis + 1];
        }
        this->slot_mask[axis] = mask;
        this->nh |= 1u << axis;
    }
    RLMAX_HD void slots_store(SweepState<NC>&) const {} // the live state is updated in place
    RLMAX_HD int dyn_at(int axis, int p) const {
        if (!(this->slot_mask[axis] & (uint64_t(1) << p))) return -1;
        for (int d = 0; d < ND; d++)
            if (slot_pos(axis, d) == p) return d;
        return -1;
    }
    RLMAX_HD int statics_below_slot(int axis, int slot) const {
        return slot - 1 - sw_popcount(this->slot_mask[axis] & ((uint64_t(1) << slot) - 1));
    }
    // Opposite edge types have different low bits, so their quantised value
    // order equals their slot order on the other, already-sorted axes.
    RLMAX_HD bool slots_overlap_2d(int a, int b, int axis) const {
        return overlap_2d(a, b, axis);
    }
    RLMAX_HD void slots_sort_down(int axis, int d) {
        const bool moving_is_max = (d & 1) != 0;
        const int moving = SW_BALL + d / 2;
        const uint16_t value = slot_value(axis, d);
        int slot = slot_pos(axis, d);
        while (slot > 1) {
            const int y = dyn_at(axis, slot - 1);
            const int k = y >= 0 ? 0 : statics_below_slot(axis, slot) - 1;
            const uint16_t p = y >= 0 ? slot_value(axis, y) : SW_STATIC_EDGE_POS[axis][k];
            if (!(value < p)) break;
            const int prev = y >= 0 ? SW_BALL + y / 2 : SW_STATIC_EDGE_H[axis][k];
            if (((p & 1) != 0) != moving_is_max && slots_overlap_2d(moving, prev, axis)) {
                if (moving_is_max) remove_pair(moving, prev);
                else add_pair(moving, prev);
            }
            if (y < 0) this->slot_mask[axis] ^= uint64_t(3) << (slot - 1);
            set_slot_pos(axis, d, --slot);
            if (y >= 0) set_slot_pos(axis, y, slot_pos(axis, y) + 1);
        }
    }
    RLMAX_HD void slots_sort_up(int axis, int d) {
        const bool moving_is_max = (d & 1) != 0;
        const int moving = SW_BALL + d / 2;
        const uint16_t value = slot_value(axis, d);
        int slot = slot_pos(axis, d);
        while (slot + 1 < NE - 1) {
            const int y = dyn_at(axis, slot + 1);
            const int k = y >= 0 ? 0 : statics_below_slot(axis, slot);
            const uint16_t p = y >= 0 ? slot_value(axis, y) : SW_STATIC_EDGE_POS[axis][k];
            if (!(value >= p)) break;
            const int next = y >= 0 ? SW_BALL + y / 2 : SW_STATIC_EDGE_H[axis][k];
            if (((p & 1) != 0) != moving_is_max && slots_overlap_2d(moving, next, axis)) {
                if (moving_is_max) add_pair(moving, next);
                else remove_pair(moving, next);
            }
            if (y < 0) this->slot_mask[axis] ^= uint64_t(3) << slot;
            set_slot_pos(axis, d, ++slot);
            if (y >= 0) set_slot_pos(axis, y, slot_pos(axis, y) - 1);
        }
    }
    RLMAX_HD void slots_update_axis(int b, int axis, uint32_t next) {
        const int emin = 2 * b, emax = 2 * b + 1;
        const uint32_t old = this->saved->edge[b][axis];
        if (old == next) return;
        const int dmin = (int)(uint16_t)next - (int)(uint16_t)old;
        const int dmax = (int)(uint16_t)(next >> 16) - (int)(uint16_t)(old >> 16);
        if (!(this->nh & (1u << axis))) prepare_slot_mask(axis);
        this->saved->edge[b][axis] = next;
        if (dmin < 0) slots_sort_down(axis, emin);
        if (dmax > 0) slots_sort_up(axis, emax);
        if (dmin > 0) slots_sort_up(axis, emin);
        if (dmax < 0) slots_sort_down(axis, emax);
    }

};

RLMAX_HD int sw_first_set(uint64_t bits) {
#if defined(__CUDA_ARCH__)
    return __ffsll((long long)bits) - 1;
#elif defined(__clang__) || defined(__GNUC__)
    return __builtin_ctzll(bits);
#else
    int index = 0;
    while (!(bits & 1u)) { bits >>= 1; ++index; }
    return index;
#endif
}

// Union-find root, (parents[root0] = root1)
RLMAX_HD int sw_root(const int* parent, int i) {
    while (parent[i] != i) i = parent[i];
    return i;
}

// btQuickSort (bullet's quickSortInternal) over `order`, keyed by `keys[order[.]]`. The two halves
// of a partition are independent, so the smaller is taken first from a bounded stack: the same order as the recursion.
RLMAX_HD void sw_hoare_sort(uint8_t* order, const uint8_t* keys, int n) {
    if (n < 2) return;
    uint8_t lo_stack[16], hi_stack[16];  // sweep manifold capacity is at most 256
    int depth = 0;
    lo_stack[0] = 0;
    hi_stack[0] = n - 1;
    depth = 1;
    while (depth > 0) {
        depth--;
        const int lo = lo_stack[depth], hi = hi_stack[depth];
        int i = lo, j = hi;
        const int pivot = keys[order[(lo + hi) / 2]];
        do {
            while (keys[order[i]] < pivot) i++;
            while (pivot < keys[order[j]]) j--;
            if (i <= j) {
                const uint8_t t = order[i];
                order[i] = order[j];
                order[j] = t;
                i++;
                j--;
            }
        } while (i <= j);
        const bool right_first = hi - i < j - lo;  // push the larger side first
        for (int side = 0; side < 2; side++) {
            const bool right = (side == 0) != right_first;
            const int a = right ? i : lo, b = right ? hi : j;
            if (a < b && depth < 16) {
                lo_stack[depth] = a;
                hi_stack[depth] = b;
                depth++;
            }
        }
    }
}

// One tick of the sweep: update the dynamic proxies, dispatch the pairs, sort the active manifolds by island. Writes the
// order of the active manifolds (sw_key) into `out` and returns their count. The state is
// rebuilt when it is not valid (after init and kickoff): the bodies added in order (the ball, then the cars) at this
// tick's boxes, no manifolds yet, so the pairs are those addHandle's axis-2 sort finds, in that order (the game adds a body at its first box; P4 lead 70: cw_turtle_goal_fall_autoroll idx 77,
// where the car meets both goal-back meshes and the recording's result needs the +x half's manifold solved after the
// -x half's, which the default-pose creation (every car at the origin, then moved) orders the other way round: 93 -> 148
// of 297 car-ticks). A state upload keeps the sweep as it is, as an edit of a live arena does; the first tick after it carries the teleport.
template <int NC, bool SlotOnly>
RLMAX_HD int sweep_step_impl(SweepState<NC>& st, const SweepIn<NC>& in, SweepKey<NC>* out, bool need_order) {
    using S = Sweep<NC, SlotOnly>;
    constexpr int NB = S::NB;
    static_assert(S::CAP <= 256, "manifold sort indices fit in a byte");
    S w;
    // The persistent caches are mutated
    // directly; their swap-with-last order is the dispatcher order.
    w.pair = st.pairs;
    w.man = st.man;
    if constexpr (SlotOnly) w.saved = &st;
    else if constexpr (NC > 2) { w.saved = nullptr; w.loaded = 7u; }
    const bool valid = (st.hdr & 1u) != 0;
    // The live slot path uses only dynamic slots; the separate full path
    // handles initialization and restoration with the complete edge lists.
    if constexpr (SlotOnly) {
        w.slots_load(st);
        w.np = (int)((st.hdr >> 8) & 255u);
        w.nm = (int)((st.hdr >> 16) & 255u);
    } else if (valid) {
        if constexpr (NC > 2 && !SlotOnly) { w.saved = &st; w.loaded = 0; }
        w.np = (int)((st.hdr >> 8) & 255u);
        w.nm = (int)((st.hdr >> 16) & 255u);
    } else {
#if defined(RLMAX_FULL_STATE)
        if (st.hdr & 8u) {
            // Restore live edges at their recorded places, then insert inactive placeholders.
            w.nh = SW_STATICS;
            for (int b = 0; b < NB; b++) w.nh += st.slot[b][0] != 255;
            const int end = 2 * w.nh + 1;
            for (int axis = 0; axis < 3; axis++) {
                bool taken[S::NE] = {};
                for (int b = 0; b < NB; b++) {
                    if (st.slot[b][0] == 255) continue;
                    const int lo = st.slot[b][2 * axis], hi = st.slot[b][2 * axis + 1];
                    taken[lo] = taken[hi] = true;
                    w.pos[axis][lo] = (uint16_t)st.edge[b][axis];
                    w.pos[axis][hi] = (uint16_t)(st.edge[b][axis] >> 16);
                    w.hnd[axis][lo] = w.hnd[axis][hi] = (uint8_t)(SW_BALL + b);
                }
                w.pos[axis][0] = 0; w.hnd[axis][0] = 0;
                w.pos[axis][end] = 0xffff; w.hnd[axis][end] = 0;
                int k = 0;
                for (int e = 1; e < end; e++) if (!taken[e]) {
                    w.pos[axis][e] = SW_STATIC_EDGE_POS[axis][k];
                    w.hnd[axis][e] = SW_STATIC_EDGE_H[axis][k++];
                }
            }
            w.index_edges();
            w.np = w.nm = 0;
            for (int b = 0; b < NB; b++) if (st.slot[b][0] == 255)
                w.add_handle(SW_BALL + b, in.qedge[b], false);
            w.np = (int)((st.hdr >> 8) & 255u);
            w.nm = (int)((st.hdr >> 16) & 255u);
        } else
#endif
        {
            w.statics_only();
            w.np = w.nm = 0;
            for (int b = 0; b < NB; b++) w.add_handle(SW_BALL + b, in.qedge[b]);
        }
    }
    // a new arena's bodies take handles in order; a demolished car gives its handle up, and takes the last one given up when it returns
    if (st.handle[0] == 0 && st.free_n == 0)
        for (int b = 0; b < NB; b++) st.handle[b] = (uint8_t)(b + 1);
    for (int b = 1; b < NB; b++)
        if (!in.response[b] && st.handle[b]) {
            st.free_handles[st.free_n++] = st.handle[b];
            st.handle[b] = 0;
        }
    for (int b = 1; b < NB; b++)
        if (in.response[b] && !st.handle[b]) st.handle[b] = st.free_handles[--st.free_n];

    // Release last tick's predictive manifolds in their original object order, by swap-with-last.
    if (valid && (st.hdr & 4u)) {
        for (int k = -1; k < NC; k++) {
            const int b = k < 0 ? 0 : 1 + visit_slot<NC>(st.predictive_visit, k);
            const int i = w.find_man(sw_key_for<NC>(0, SW_BALL + b));
            if (i >= 0) w.man[i] = w.man[--w.nm];
        }
    }
    if (in.predictive) {
        for (int k = -1; k < NC; k++) {
            const int b = k < 0 ? 0 : 1 + visit_slot<NC>(in.visit, k);
            if ((in.predictive >> b & 1u) && w.nm < S::CAP) w.man[w.nm++] = sw_key_for<NC>(0, SW_BALL + b);
        }
        st.predictive_visit = in.visit;
    }

    // updateAabbs: the ball, then the cars
    for (int b = 0; b < NB; b++)
        for (int axis = 0; axis < 3; axis++) {
            if constexpr (SlotOnly) w.slots_update_axis(b, axis, in.qedge[b][axis]);
            else w.update_axis(SW_BALL + b, axis, in.qedge[b][axis]);
        }

    // dispatchAllCollisionPairs in pair-cache order
    uint64_t dynamic_pairs = 0;
    w.act.clear();
    for (int i = 0; i < w.nm; i++) w.act.set(i, sw_lo_for<NC>(w.man[i]) == 0);
    for (int i = 0; i < w.np; i++) {
        const SweepKey<NC> k = w.pair[i];
        const int lo = sw_lo_for<NC>(k), hi = sw_hi_for<NC>(k);
        const bool lo_static = lo <= SW_STATICS;
        if constexpr (S::PAIR_CAP <= 64) if (!lo_static) dynamic_pairs |= uint64_t(1) << i;
        const int bl = lo - SW_BALL, bh = hi - SW_BALL;  // dynamic indices (negative for a static)
        const bool act_lo = !lo_static && in.active[bl], act_hi = in.active[bh];
        // a ball-static pair keeps its manifold while the ball sleeps, and a sleeping ball makes no new one: it comes when the
        // ball wakes, after any ball-car manifold made meanwhile
        const bool ball_static = lo_static && bh == 0;
        if ((!ball_static && !act_lo && !act_hi) || (!lo_static && !in.response[bl]) || !in.response[bh]) continue;
        // a car is a compound: its child box must overlap the other shape's box (a static's raw box, the ball's box)
        // or the child algorithm and its manifold go; so for two cars the other's child box does
        // not gate the compound pair, truth sim.aabb_gates ProcessChildShape destroys the child algorithm on separation)
        if (bh >= 1) {
            bool gate = true;
            if (lo_static) gate = sw_static_gate(lo, in.child_min[bh - 1], in.child_max[bh - 1]);
            else if (bl == 0) gate = aabb_overlap(in.child_min[bh - 1], in.child_max[bh - 1], in.ball_min, in.ball_max);
            else gate = aabb_overlap(in.child_min[bh - 1], in.child_max[bh - 1], in.child_min[bl - 1], in.child_max[bl - 1]);
            if (!gate) {
                w.release(k);
                continue;
            }
        }
        // Pair and manifold arrays often retain the same order; check the
        // corresponding slot first, then retain the exact linear-search fallback.
        int m = i < w.nm && w.man[i] == k ? i : w.find_man(k);
        // a held plane manifold is kept in place, and a new one is made only once the ball wakes
        if (m < 0 && ball_static && !in.active[0] && !(lo > 16 && lo <= 20 && (in.ball_planes_held >> (lo - 17) & 1u))) continue;
        if (m < 0 && bh >= 1 && lo_static && !sw_static_local_gate<NC>(in, lo, bh)) continue;
        if (m < 0 && bh >= 1 && !lo_static) {
            // The compound query uses the lower-handle car's frame.
            const bool swap = bl >= 1 && st.handle[bh] < st.handle[bl];
            const int a = bl >= 1 && !swap ? bl : bh;
            const int b = bl >= 1 ? (swap ? bl : bh) : 0;
            if (!sw_local_overlap<NC>(in, a, b)) continue;
        }
        if (m < 0 && w.nm < S::CAP) {
            m = w.nm++;
            w.man[m] = k;
        }
        if (m >= 0) w.act.set(m, true);
    }

    // The sorted active order is used only by the solver. If no known contact
    // needs it, an extant car-car manifold (or restored predictive entry) may
    // still produce a row after this sweep, so retain the sort for those cases.
    if (!need_order) {
        for (int i = 0; i < w.nm; i++) {
            const int lo = sw_lo_for<NC>(w.man[i]);
            if (lo == 0 || lo > SW_BALL) { need_order = true; break; }
        }
        if (!need_order) {
            if constexpr (SlotOnly) w.slots_store(st);
            else w.store(st);
            st.hdr = 1u | (in.predictive ? 4u : 0u) | ((uint32_t)w.np << 8) | ((uint32_t)w.nm << 16);
            return 0;
        }
    }

    // islands: the dynamic pairs in pair-cache order unite (proxy0 into proxy1); a manifold's key is its dynamic body's
    // root
    // (the unions run over the bodies' places in the sweep's object list: the ball, then the
    // cars in the game's visit order, a demolished car removed; the island id is the root's place)
    int tag[NB];
    tag[0] = 0;
    for (int k = 0, t = 1; k < NC; k++) {
        const int ci = visit_slot<NC>(in.visit, k);
        tag[1 + ci] = (in.visit >> (24 + ci) & 1u) ? -1 : t++;
    }
    int parent[NB];
    for (int b = 0; b < NB; b++) parent[b] = b;
    for (int index = 0; index < w.np; index++) {
        int i = index;
        if constexpr (S::PAIR_CAP <= 64) {
            if (!dynamic_pairs) break;
            i = sw_first_set(dynamic_pairs);
            dynamic_pairs &= dynamic_pairs - 1;
        }
        const int lo = sw_lo_for<NC>(w.pair[i]), hi = sw_hi_for<NC>(w.pair[i]);
        if (lo < SW_BALL) continue;
        const int x = lo - SW_BALL, y = hi - SW_BALL;
        if (!in.response[x] || !in.response[y] || tag[x] < 0 || tag[y] < 0) continue;
        const bool x_first = st.handle[x] <= st.handle[y];  // proxy0 is the lower handle
        const int ra = sw_root(parent, tag[x_first ? x : y]), rb = sw_root(parent, tag[x_first ? y : x]);
        if (ra != rb) parent[ra] = rb;
    }
    uint8_t order[S::CAP], keys[S::CAP];
    int n = 0;
    for (int i = 0; i < w.nm; i++) {  // ascending manifold index (active indices sorted, then the island sort)
        if (!w.act.get(i)) continue;
        const int lo = sw_lo_for<NC>(w.man[i]), hi = sw_hi_for<NC>(w.man[i]);
        const int tg = tag[(lo >= SW_BALL ? lo : hi) - SW_BALL];
        keys[i] = (uint8_t)(tg < 0 ? 0 : sw_root(parent, tg));
        order[n++] = (uint8_t)i;
    }
    sw_hoare_sort(order, keys, n);
    for (int i = 0; i < n; i++) out[i] = w.man[order[i]];

    // store
    if constexpr (SlotOnly) w.slots_store(st);
    else w.store(st);
    st.hdr = 1u | (in.predictive ? 4u : 0u) | ((uint32_t)w.np << 8) | ((uint32_t)w.nm << 16);
    return n;
}

// Initial and restored lists are rare; keep their full arrays out of the
// live tick frame for standard 1v1, 2v2 and 3v3 arenas.
template <int NC>
RLMAX_COLD int sweep_rebuild_step(SweepState<NC>& st, const SweepIn<NC>& in, SweepKey<NC>* out, bool need_order) {
    return sweep_step_impl<NC, false>(st, in, out, need_order);
}
template <int NC>
RLMAX_HD int sweep_step(SweepState<NC>& st, const SweepIn<NC>& in, SweepKey<NC>* out, bool need_order = true) {
    if constexpr (NC <= 2 || NC == 4 || NC == 6) {
        if (st.hdr & 1u) return sweep_step_impl<NC, true>(st, in, out, need_order);
        return sweep_rebuild_step<NC>(st, in, out, need_order);
    } else {
        return sweep_step_impl<NC, false>(st, in, out, need_order);
    }
}

// A round reset removes every body, then adds them all back from the free list: last released, first reused, so the order turns over (the suite's
// recorded handles: 23, 24, 25 then 25, 24, 23 and back at each goal).
template <int NC>
RLMAX_HD void sweep_round_reset(SweepState<NC>& st) {
    constexpr int NB = 1 + NC;
    if (st.handle[0] == 0 && st.free_n == 0) return;  // none given yet
    for (int b = 0; b < NB; b++)
        if (st.handle[b]) {
            st.free_handles[st.free_n++] = st.handle[b];
            st.handle[b] = 0;
        }
    for (int b = 0; b < NB; b++) st.handle[b] = st.free_handles[--st.free_n];
}

// The car pairs that hold a manifold in the sweep's dispatcher array: bit p for the pair (i < j) in the loop order of
// tick.h's pair manifolds. A sweep that is not valid yet is rebuilt at the next step with no manifolds.
template <int NC>
RLMAX_HD uint32_t sweep_car_pairs(const SweepState<NC>& st) {
    static_assert(NC * (NC - 1) / 2 <= 32, "one bit per car pair");
    uint32_t mask = 0;
    const int nm = (int)((st.hdr >> 16) & 255u);
    for (int q = 0; q < sweep_man_cap<NC>(); q++) {
        if (q >= nm) break;
        const int lo = sw_lo_for<NC>(st.man[q]) - SW_BALL - 1, hi = sw_hi_for<NC>(st.man[q]) - SW_BALL - 1;
        if (lo < 0) continue;  // a static or the ball against a car
        mask |= 1u << (lo * NC - lo * (lo + 1) / 2 + hi - lo - 1);
    }
    return mask;
}

// The tick's boxes for the sweep: each body's proxy AABB is the box now joined with the box one tick on (the predicted
// transform: the shape's AABB at the world and at the interpolation transform, each widened by the
// contact breaking threshold), quantised; the cars' child boxes and the ball's box gate the dispatch. Out of line, so the
// tick's registers and frame stay as they were.
template <int NC>
RLMAX_COLD int sweep_tick(SweepState<NC>& st, const Body* bodies, const V3* predict_lin, const V3* predict_ang, const CarGeo* geo, float ball_radius, uint32_t visit, uint32_t predictive, SweepKey<NC>* out, bool need_order = true, uint32_t ball_planes_held = 0) {
    constexpr int NB = 1 + NC;
    SweepIn<NC> swin;
    swin.bodies = bodies;
    swin.geo = geo;
    swin.visit = visit;
    swin.predictive = predictive;
    swin.ball_planes_held = ball_planes_held;
    const float cbt = c::CONTACT_BREAKING_THRESHOLD;
    for (int b = 0; b < NB; b++) {
        const Body& rb = bodies[b];
        V3 nmn, nmx, pmn, pmx;
        if (b == 0) {
            const V3 e = v3s(ball_radius + c::SPHERE_BOUND_PAD_BT);
            nmn = rb.pos - e;
            nmx = rb.pos + e;
            const V3 pp = rb.pos + predict_lin[0] * c::DT;
            pmn = pp - e;
            pmx = pp + e;
            swin.ball_min = nmn;
            swin.ball_max = nmx;
            swin.ball_e = ball_radius + c::SPHERE_BOUND_PAD_BT;
        } else {
            const Obb now = car_obb(rb, geo[b - 1]);
            obb_aabb(now, &nmn, &nmx);
            swin.child_min[b - 1] = nmn;
            swin.child_max[b - 1] = nmx;
            // the box one tick on, in the game's arithmetic: the origin from the uu velocity, the basis by the tick's integration
            Body p;
            p.pos = rb.pos;
            p.r = rb.r;
            integrate_game(p, v3s(0.0f), predict_ang[b], c::DT);
            const V3 pp = rb.pos + (predict_lin[b] * c::BT_TO_UU) * (c::UU_TO_BT * c::DT);
            const V3 off = geo[b - 1].hitbox_off;
            const V3 pc = v3(((pp.x + p.r.c0.x * off.x) + p.r.c1.x * off.y) + p.r.c2.x * off.z, ((pp.y + p.r.c0.y * off.x) + p.r.c1.y * off.y) + p.r.c2.y * off.z, ((pp.z + p.r.c0.z * off.x) + p.r.c1.z * off.y) + p.r.c2.z * off.z);
            obb_aabb(Obb{pc, p.r, geo[b - 1].hitbox_half, geo[b - 1].margin}, &pmn, &pmx);
        }
        const V3 cb = v3s(cbt);
        nmn = nmn - cb;
        nmx = nmx + cb;
        pmn = pmn - cb;
        pmx = pmx + cb;
        const V3 lo = v3(fmin_(nmn.x, pmn.x), fmin_(nmn.y, pmn.y), fmin_(nmn.z, pmn.z));
        const V3 hi = v3(fmax_(nmx.x, pmx.x), fmax_(nmx.y, pmx.y), fmax_(nmx.z, pmx.z));
        st.box[b][0] = lo.x; st.box[b][1] = lo.y; st.box[b][2] = lo.z;
        st.box[b][3] = hi.x; st.box[b][4] = hi.y; st.box[b][5] = hi.z;
        for (int a = 0; a < 3; a++) {
            const uint16_t qmin = sw_quant(vget(lo, a), 0);
            const uint16_t qmax = sw_quant(vget(hi, a), 1);
            swin.qedge[b][a] = (uint32_t)qmin | ((uint32_t)qmax << 16);
        }
        swin.active[b] = rb.active;
        swin.response[b] = rb.contact_response;
    }
    const int n = sweep_step<NC>(st, swin, out, need_order);
    st.hdr |= 2u;
    return n;
}

}  // namespace rlmax
