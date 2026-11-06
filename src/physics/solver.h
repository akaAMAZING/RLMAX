// RLMAX solver: Bullet's sequential impulse as the game configures it, with fixed iteration counts. The earlier model stops the velocity iterations once the maximum
// squared residual is exactly zero. Split rows remain active until an entire iteration applies no impulse: another
// row can disturb a row whose previous residual was zero.
//
// Rows store only what the solve produces or what depends on the pre-solve velocities (the accumulated impulses and the
// right-hand sides) plus the contact geometry (lever arms, normal; a friction row its axis). The Jacobians and the
// inverse effective mass are recomputed at every resolve with the same expressions as at setup, so every value the
// solve uses is the same bit pattern as a stored copy would be, and the row arrays stay small.
#pragma once

#include "contacts.h"

namespace rlmax {

struct SBody {
    V3 lin, ang;          // velocity at the start of the solve
    V3 ext_lin, ext_ang;  // accumulated (external) impulses this tick
    V3 dlin, dang;        // solver deltas
    V3 push, turn;        // split-impulse pseudo velocities
    M3 inv_i;             // world inverse inertia
    float inv_mass;
    bool dynamic;         // false: the fixed body (static world, sleeping or demolished)
    uint8_t handle;  // the body's broadphase handle (SweepState)
};

// A contact row (also a split-only row and a synthetic averaged row). Its direction on A is n, on B -n (exactly).
struct CRow {
    int a, b;            // body indices; -1 = fixed
    V3 rp1, rp2, n;      // lever arms from the bodies' positions, normal on B
    float rhs, rhs_pen, applied, applied_push, friction;
    bool split_only;
};

// A friction row: its contact row (bodies, lever arms, friction coefficient) and its axis.
struct FRow {
    int parent;
    V3 axis;
    float rhs, applied;
};

// A row's Jacobian terms, recomputed where they are used.
struct Jac {
    V3 rxn1, rxn2, ang_a, ang_b;
    float jac_inv;
};

struct ContactPoint {
    int8_t a, b;    // body indices (b = -1: static)
    bool special;   // ball vs static: split row only, then averaged into one synthetic row per body
    bool ok;
    V3 pa, pb, n;   // world points on A and B, normal on B
    float d;
    float friction, restitution;
};
static_assert(sizeof(ContactPoint) == 52, "a contact point packs its small fields");

template <int MAX_ROWS, int MAX_FRICTION, int NB>
struct Solver {
    SBody bodies[NB];
    CRow rows[MAX_ROWS];
    FRow fric[MAX_FRICTION];
    int num_rows = 0, num_fric = 0;
    // Only ball-world contacts are special (tick.h): all use body 0.
    V3 sp_normal;
    float sp_lever, sp_friction, sp_restitution;
    int sp_count;

    RLMAX_HD void reset() {
        num_rows = num_fric = 0;
        sp_count = 0;
    }

    RLMAX_HD V3 vel_no_delta(int bi, V3 rel) const {
        if (bi < 0) return v3s(0.0f);
        const SBody& s = bodies[bi];
        if (!s.dynamic) return v3s(0.0f);
        return (s.lin + s.ext_lin) + cross(s.ang + s.ext_ang, rel);
    }

    RLMAX_HD bool dyn(int bi) const { return bi >= 0 && bodies[bi].dynamic; }

    // Bullet's plane_space_1
    RLMAX_HD static V3 plane_space(V3 n) {
        if (absf(n.z) > 0.70710678118654752f) {
            float a = n.y * n.y + n.z * n.z;
            float k = 1.0f / sqrtf(a);
            return v3(0.0f, -n.z * k, n.y * k);
        }
        float a = n.x * n.x + n.y * n.y;
        float k = 1.0f / sqrtf(a);
        return v3(-n.y * k, n.x * k, 0.0f);
    }

    RLMAX_HD V3 lat_friction_dir(int a, int b, V3 n, V3 rp1, V3 rp2) const {
        V3 vel = vel_no_delta(a, rp1) - vel_no_delta(b, rp2);
        float rel = dot(n, vel);
        V3 lat = vel - n * rel;
        float l2 = len2(lat);
        if (l2 > FLT_EPS) return lat * (1.0f / sqrtf(l2));
        return plane_space(n);
    }

    // setup_contact_constraint / setup_split_contact_constraint: the Jacobian of a contact row along its normal
    RLMAX_HD Jac contact_jac(const CRow& r) const {
        Jac j{};
        j.rxn1 = j.rxn2 = j.ang_a = j.ang_b = v3s(0.0f);
        float denom0 = 0.0f, denom1 = 0.0f;
        if (dyn(r.a)) {
            const SBody& s = bodies[r.a];
            V3 ta = cross(r.rp1, r.n);
            j.ang_a = mul_t(s.inv_i, ta);
            denom0 = s.inv_mass + dot(r.n, cross(j.ang_a, r.rp1));
            j.rxn1 = ta;
        }
        if (dyn(r.b)) {
            const SBody& s = bodies[r.b];
            V3 ta = cross(r.rp2, r.n);
            j.ang_b = mul_t(s.inv_i, -ta);
            denom1 = s.inv_mass + dot(r.n, cross(-j.ang_b, r.rp2));
            j.rxn2 = -ta;
        }
        j.jac_inv = 1.0f / (denom0 + denom1);
        return j;
    }

    // setup_friction_constraint: the Jacobian of a friction row along `axis` at its contact row's points
    RLMAX_HD Jac friction_jac(const CRow& p, V3 axis) const {
        Jac j{};
        j.rxn1 = j.rxn2 = j.ang_a = j.ang_b = v3s(0.0f);
        float denom1 = 0.0f, denom2 = 0.0f;
        if (dyn(p.a)) {
            const SBody& s = bodies[p.a];
            j.rxn1 = cross(p.rp1, axis);
            j.ang_a = mul_t(s.inv_i, j.rxn1);
            denom1 = s.inv_mass + dot(axis, cross(j.ang_a, p.rp1));
        }
        if (dyn(p.b)) {
            const SBody& s = bodies[p.b];
            j.rxn2 = cross(p.rp2, -axis);
            j.ang_b = mul_t(s.inv_i, j.rxn2);
            denom2 = s.inv_mass + dot(axis, cross(-j.ang_b, p.rp2));
        }
        j.jac_inv = 1.0f / (denom1 + denom2);
        return j;
    }

    // setup_contact_constraint (no warm start: the game's applied impulses are always zero)
    RLMAX_HD CRow contact_row(int a, int b, V3 rp1, V3 rp2, V3 n, float d, float friction, float restitution) const {
        CRow r{};
        r.a = a;
        r.b = b;
        r.rp1 = rp1;
        r.rp2 = rp2;
        r.n = n;
        r.friction = friction;
        r.applied = r.applied_push = 0.0f;
        r.split_only = false;
        const Jac j = contact_jac(r);
        float v1n = 0.0f, v2n = 0.0f;
        V3 vel0 = v3s(0.0f), vel1 = v3s(0.0f);
        if (dyn(a)) {
            const SBody& s = bodies[a];
            vel0 = s.lin + cross(s.ang, rp1);
            v1n = dot(n, s.lin + s.ext_lin) + dot(j.rxn1, s.ang + s.ext_ang);
        }
        if (dyn(b)) {
            const SBody& s = bodies[b];
            vel1 = s.lin + cross(s.ang, rp2);
            v2n = dot(-n, s.lin + s.ext_lin) + dot(j.rxn2, s.ang + s.ext_ang);
        }
        float rel_vel0 = dot(n, vel0 - vel1);
        float rest = fmax_(restitution * -rel_vel0, 0.0f);
        float pos_err = d > 0.0f ? 0.0f : -d * c::ERP2 / c::DT;
        float vel_err = rest - (v1n + v2n);
        r.rhs = vel_err * j.jac_inv;
        r.rhs_pen = pos_err * j.jac_inv;
        return r;
    }

    // setup_split_contact_constraint
    RLMAX_HD CRow split_row(int a, int b, V3 rp1, V3 rp2, V3 n, float d) const {
        CRow r{};
        r.a = a;
        r.b = b;
        r.rp1 = rp1;
        r.rp2 = rp2;
        r.n = n;
        r.friction = 0.0f;
        r.rhs = 0.0f;
        r.applied = r.applied_push = 0.0f;
        r.split_only = true;
        const Jac j = contact_jac(r);
        float pos_err = d > 0.0f ? 0.0f : -d * c::ERP2 / c::DT;
        r.rhs_pen = pos_err * j.jac_inv;
        return r;
    }

    // setup_friction_constraint
    RLMAX_HD FRow friction_row(int parent, V3 axis) const {
        const CRow& p = rows[parent];
        FRow f{};
        f.parent = parent;
        f.axis = axis;
        f.applied = 0.0f;
        const Jac j = friction_jac(p, axis);
        float v1n = 0.0f, v2n = 0.0f;
        if (dyn(p.a)) {
            const SBody& s = bodies[p.a];
            v1n = dot(axis, s.lin + s.ext_lin) + dot(j.rxn1, s.ang);
        }
        if (dyn(p.b)) {
            const SBody& s = bodies[p.b];
            v2n = dot(-axis, s.lin + s.ext_lin) + dot(j.rxn2, s.ang);
        }
        f.rhs = -(v1n + v2n) * j.jac_inv;
        return f;
    }

    // One manifold point. Special points (ball vs static) add a split row and accumulate into the body's synthetic row.
    RLMAX_HD void add_point(const ContactPoint& cp, V3 pos_a, V3 pos_b) {
        if (!cp.ok) return;
        V3 rp1 = cp.pa - pos_a;
        V3 rp2 = cp.pb - pos_b;
        if (cp.special && quirk::BALL_STATIC_AVERAGED_ROW) {
            if (num_rows < MAX_ROWS) rows[num_rows++] = split_row(cp.a, cp.b, rp1, rp2, cp.n, cp.d);
            int body = dyn(cp.a) ? cp.a : cp.b;
            V3 lever = dyn(cp.a) ? rp1 : rp2;
            if (body >= 0) {
                if (sp_count == 0) {
                    sp_normal = v3s(0.0f);
                    sp_lever = sp_friction = sp_restitution = 0.0f;
                }
                sp_normal += cp.n;
                sp_lever += len(lever);
                sp_friction += cp.friction;
                sp_restitution += cp.restitution;
                sp_count += 1;
            }
            return;
        }
        if (num_rows >= MAX_ROWS || num_fric >= MAX_FRICTION) return;
        int parent = num_rows;
        V3 lat = lat_friction_dir(cp.a, cp.b, cp.n, rp1, rp2);
        rows[num_rows++] = contact_row(cp.a, cp.b, rp1, rp2, cp.n, cp.d, cp.friction, cp.restitution);
        fric[num_fric++] = friction_row(parent, lat);
    }

    // emit_special_synthetics: one averaged row (plus friction) per body with special points, after all manifolds.
    RLMAX_HD void emit_synthetics() {
        if (sp_count == 0 || num_rows >= MAX_ROWS || num_fric >= MAX_FRICTION) return;
        const int bi = 0;
        // a lone point keeps its own normal and values; several are averaged, the normal summed then normalised
        V3 mean_n = sp_normal;
        float mean_lever = sp_lever, mu = sp_friction, e = sp_restitution;
        if (sp_count > 1) {
            const float inv = 1.0f / (float)sp_count;
            const float l2 = len2(mean_n);
            mean_n = l2 >= FLT_EPS * FLT_EPS ? mean_n * (1.0f / sqrtf(l2)) : v3(1.0f, 0.0f, 0.0f);
            mean_lever = inv * mean_lever;
            mu = inv * mu;
            e = inv * e;
        }
        V3 rp1 = mean_n * -mean_lever;
        V3 rp2 = v3s(0.0f);
        int parent = num_rows;
        rows[num_rows++] = contact_row(bi, -1, rp1, rp2, mean_n, 0.0f, mu, e);
        V3 lat = lat_friction_dir(bi, -1, mean_n, rp1, rp2);
        fric[num_fric++] = friction_row(parent, lat);
    }

    // resolve_single_constraint_row_{lower_limit,generic} along n (B: -n); returns the residual. Contact rows: lower
    // 0 and no upper limit; friction rows: minus and plus friction times the parent's impulse.
    // A car pair's row takes the term of the game's body0, the lower broadphase handle, first. A ball row (B = the ball, body 0) and a world row keep A's term first.
    RLMAX_HD bool b_first(int a, int b) const { return b > 0 && bodies[b].handle < bodies[a].handle; }

    RLMAX_HD float resolve(int a, int b, V3 n, const Jac& j, float rhs, float& applied, float lower, float upper,
                           bool upper_limit) {
        const bool active_a = dyn(a), active_b = dyn(b);
        float delta = rhs;
        float dv1 = 0.0f, dv2 = 0.0f;
        if (active_a) dv1 = dot(n, bodies[a].dlin) + dot(j.rxn1, bodies[a].dang);
        if (active_b) dv2 = dot(-n, bodies[b].dlin) + dot(j.rxn2, bodies[b].dang);
        const bool second_first = b_first(a, b);
        delta -= (second_first ? dv2 : dv1) * j.jac_inv;
        delta -= (second_first ? dv1 : dv2) * j.jac_inv;
        float sum = applied + delta;
        if (sum < lower) {
            delta = lower - applied;
            applied = lower;
        } else if (upper_limit && sum > upper) {
            delta = upper - applied;
            applied = upper;
        } else {
            applied = sum;
        }
        if (active_a) {
            SBody& s = bodies[a];
            s.dlin += n * s.inv_mass * delta;
            s.dang += j.ang_a * delta;
        }
        if (active_b) {
            SBody& s = bodies[b];
            s.dlin += -n * s.inv_mass * delta;
            s.dang += j.ang_b * delta;
        }
        return delta / j.jac_inv;
    }

    RLMAX_HD float resolve_split_jac(CRow& r, const Jac& j) {
        const bool active_a = dyn(r.a), active_b = dyn(r.b);
        float delta = r.rhs_pen;
        float dv1 = 0.0f, dv2 = 0.0f;
        if (active_a) dv1 = dot(r.n, bodies[r.a].push) + dot(j.rxn1, bodies[r.a].turn);
        if (active_b) dv2 = dot(-r.n, bodies[r.b].push) + dot(j.rxn2, bodies[r.b].turn);
        const bool second_first = b_first(r.a, r.b);
        delta -= (second_first ? dv2 : dv1) * j.jac_inv;
        delta -= (second_first ? dv1 : dv2) * j.jac_inv;
        float sum = r.applied_push + delta;
        if (sum < 0.0f) {
            delta = 0.0f - r.applied_push;
            r.applied_push = 0.0f;
        } else {
            r.applied_push = sum;
        }
        if (delta == 0.0f) return 0.0f;
        if (active_a) {
            SBody& s = bodies[r.a];
            s.push += r.n * s.inv_mass * delta;
            s.turn += j.ang_a * delta;
        }
        if (active_b) {
            SBody& s = bodies[r.b];
            s.push += -r.n * s.inv_mass * delta;
            s.turn += j.ang_b * delta;
        }
        return delta;
    }

    RLMAX_HD float resolve_split(CRow& r) {
        if (r.rhs_pen == 0.0f) return 0.0f;
        const Jac j = contact_jac(r);
        return resolve_split_jac(r, j);
    }

    // Row geometry and inverse inertia do not change during the solve.
    // Keep the one-friction specialization out of the empty and multi-friction paths.
    RLMAX_COLD void solve_one() {
        FRow& f = fric[0];
        const int parent = f.parent;
        CRow& r = rows[parent];
        const Jac normal = contact_jac(r);
        if (num_rows == 1) {
            if (r.rhs_pen != 0.0f) {
                for (int it = 0; it < c::SOLVER_ITERATIONS; ++it)
                    if (resolve_split_jac(r, normal) == 0.0f) break;
            }
        } else {
            for (int it = 0; it < c::SOLVER_ITERATIONS; ++it) {
                bool changed = false;
                for (int i = 0; i < num_rows; ++i) {
                    if (i == parent) {
                        if (r.rhs_pen != 0.0f) changed |= resolve_split_jac(r, normal) != 0.0f;
                    } else {
                        changed |= resolve_split(rows[i]) != 0.0f;
                    }
                }
                if (!changed) break;
            }
        }
        Jac tangent{};
        bool tangent_ready = false;
        for (int it = 0; it < c::SOLVER_ITERATIONS; ++it) {
            float res = resolve(r.a, r.b, r.n, normal, r.rhs, r.applied, 0.0f, 0.0f, false);
            float lsr = fmax_(res * res, 0.0f);
            const float total = r.applied;
            if (!(total <= 0.0f)) {
                const float limit = r.friction * total;
                if (!tangent_ready) { tangent = friction_jac(r, f.axis); tangent_ready = true; }
                res = resolve(r.a, r.b, f.axis, tangent, f.rhs, f.applied, -limit, limit, true);
                lsr = fmax_(res * res, lsr);
            }
            if (quirk::SOLVER_EARLY_EXIT && lsr == 0.0f) break;
        }
    }

    RLMAX_HD void solve() {
        if (num_rows == 0) return;  // every loop below is empty: friction rows need a parent, and the early exit changes nothing
        // Each non-split row is created together with exactly one friction row.
        if (num_fric == 1) { solve_one(); return; }
        // A zero row can become active again after a later row changes either body.
        for (int it = 0; it < c::SOLVER_ITERATIONS; it++) {
            bool changed = false;
            for (int i = 0; i < num_rows; i++) changed |= resolve_split(rows[i]) != 0.0f;
            if (!changed) break;
        }
        // velocity pass, fixed iterations; the residual test only reproduces the earlier model's early exit
        bool done = false;
        for (int it = 0; it < c::SOLVER_ITERATIONS; it++) {
            if (done) break;
            float lsr = 0.0f;
            for (int i = 0; i < num_rows; i++) {
                if (rows[i].split_only) continue;
                CRow& r = rows[i];
                const Jac j = contact_jac(r);
                float res = resolve(r.a, r.b, r.n, j, r.rhs, r.applied, 0.0f, 0.0f, false);
                lsr = fmax_(res * res, lsr);
            }
            for (int i = 0; i < num_fric; i++) {
                FRow& f = fric[i];
                const CRow& p = rows[f.parent];
                float total = p.applied;
                if (total <= 0.0f) continue;
                float limit = p.friction * total;
                const Jac j = friction_jac(p, f.axis);
                float res = resolve(p.a, p.b, f.axis, j, f.rhs, f.applied, -limit, limit, true);
                lsr = fmax_(res * res, lsr);
            }
            done = quirk::SOLVER_EARLY_EXIT && lsr == 0.0f;
        }
    }
};

}  // namespace rlmax
