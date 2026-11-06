// Rigid-body working state for one tick, and the operations Bullet performs on it.
#pragma once

#include "consts.h"
#include "impulse_log.h"
#include "full_state.h"

namespace rlmax {

struct Body {
    V3 pos;
    Q rot;
    M3 r;        // rotation matrix of `rot`
    V3 vel, ang;
    V3 acc_lin, acc_ang;  // accumulated impulses, applied by the solver
    V3 pos_uu;  // the position the game reads in uu, likewise
    V3 vel_uu;  // the velocity the game reads in uu: the codec's value at tick start, not BT x 50
    V3 torque;  // cars: the tick's torques, BT (Bullet's m_totalTorque)
    V3 force;  // cars: the tick's central forces, BT (Bullet's m_totalForce); made into acc_lin once, with gravity
    M3 inv_i;    // world inverse inertia
    V3 inv_i_local;
    float mass, inv_mass;
    bool active;            // simulated this tick (not demolished, not sleeping)
    bool contact_response;  // takes part in collisions (not demolished)
};

RLMAX_HD M3 world_inv_inertia(const M3& r, V3 inv_local) {
    // R * diag(inv_local) * R^T
    M3 rt = transpose(r);
    M3 scaled{rt.c0 * inv_local, rt.c1 * inv_local, rt.c2 * inv_local};
    return r * scaled;
}

RLMAX_HD M3 bullet_set_rotation(Q q);

// The basis of a stored quaternion is Bullet's setRotation of it (s = 2 / |q|^2): the basis the integration wrote, which the
// next tick-start quantise reads (the game keeps the basis columns; q_to_m3's 2q product rounds differently, and the
// codec step then fell the other way on partycannon 849: z 5.4e-6 off).
RLMAX_HD void body_set_rot(Body& b, Q q) {
    b.rot = q;
    b.r = bullet_set_rotation(q);
    b.inv_i = world_inv_inertia(b.r, b.inv_i_local);
}

enum ImpulseKind { IMP_LINEAR, IMP_LINEAR_REL, IMP_ANGULAR };

// add_impulse(impulse, massed, accum)
RLMAX_HD void add_impulse(Body& b, ImpulseKind kind, V3 v, V3 rel, bool massed, bool accum) {
    V3 lin = v3s(0.0f), angi = v3s(0.0f);
    float ms = massed ? b.inv_mass : 1.0f;
    if (kind == IMP_LINEAR) {
        lin = v * ms;
    } else if (kind == IMP_LINEAR_REL) {
        lin = v * ms;
        angi = mul_t(b.inv_i, cross(rel, v));
        if (!massed) angi = angi * b.mass;
    } else {
        angi = v * ms;
    }
    RLMAX_IMP_LAST(lin, angi, accum);
    if (accum) {
        b.acc_lin += lin;
        b.acc_ang += angi;
    } else {
        b.vel += lin;
        b.ang += angi;
    }
}
RLMAX_HD void add_lin(Body& b, V3 v, bool massed, bool accum) { add_impulse(b, IMP_LINEAR, v, v3s(0), massed, accum); }
RLMAX_HD void add_rel(Body& b, V3 v, V3 rel, bool massed, bool accum) { add_impulse(b, IMP_LINEAR_REL, v, rel, massed, accum); }
RLMAX_HD void add_ang(Body& b, V3 v, bool massed, bool accum) { add_impulse(b, IMP_ANGULAR, v, v3s(0), massed, accum); }

RLMAX_HD void limit_vels(Body& b, float max_lin, float max_ang) {
    if (len2(b.vel) > max_lin * max_lin) b.vel = normalize_or_zero(b.vel) * max_lin;
    if (len2(b.ang) > max_ang * max_ang) b.ang = normalize_or_zero(b.ang) * max_ang;
}

RLMAX_HD V3 vel_at(const Body& b, V3 rel) { return b.vel + cross(b.ang, rel); }

// ---- quantisation -----------------------------------------------------------------------------

// times the reciprocal, not divided: scale 100 is not a power of two, so the two differ in the last bit on a share of
// positions
RLMAX_HD float q_pos(float x, float scale) { return floorf(x * scale + 0.5f) * (1.0f / scale); }
RLMAX_HD float sign3(float x) { return x == 0.0f ? 0.0f : (x < 0.0f ? -1.0f : 1.0f); }
RLMAX_HD float q_vel(float x, float scale) {
    float inv = 1.0f / scale;
    float r = (float)(int)(x * scale) * inv;
    return r + sign3(r) * (0.1f * inv);
}
// Bullet's btMatrix3x3::getRotation on a basis whose columns are c0..c2 (m[row][col] = c[col][row]).
RLMAX_HD Q bullet_get_rotation(const M3& r) {
    const float m[3][3] = {{r.c0.x, r.c1.x, r.c2.x}, {r.c0.y, r.c1.y, r.c2.y}, {r.c0.z, r.c1.z, r.c2.z}};
    float q[4];
    const float trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f);
        q[3] = s * 0.5f;
        s = 0.5f / s;
        q[0] = (m[2][1] - m[1][2]) * s;
        q[1] = (m[0][2] - m[2][0]) * s;
        q[2] = (m[1][0] - m[0][1]) * s;
    } else {
        const int i = m[0][0] < m[1][1] ? (m[1][1] < m[2][2] ? 2 : 1) : (m[0][0] < m[2][2] ? 2 : 0);
        const int j = (i + 1) % 3, k = (i + 2) % 3;
        float s = sqrtf(m[i][i] - m[j][j] - m[k][k] + 1.0f);
        q[i] = s * 0.5f;
        s = 0.5f / s;
        q[3] = (m[k][j] - m[j][k]) * s;
        q[j] = (m[j][i] + m[i][j]) * s;
        q[k] = (m[k][i] + m[i][k]) * s;
    }
    return Q{q[0], q[1], q[2], q[3]};
}

// Bullet's btMatrix3x3::setRotation (s = 2 / |q|^2), as columns.
RLMAX_HD M3 bullet_set_rotation(Q q) {
    const float d = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    const float s = 2.0f / d;
    const float xs = q.x * s, ys = q.y * s, zs = q.z * s;
    const float wx = q.w * xs, wy = q.w * ys, wz = q.w * zs;
    const float xx = q.x * xs, xy = q.x * ys, xz = q.x * zs;
    const float yy = q.y * ys, yz = q.y * zs, zz = q.z * zs;
    return M3{v3(1.0f - (yy + zz), xy + wz, xz - wy), v3(xy - wz, 1.0f - (xx + zz), yz + wx),
              v3(xz + wy, yz - wx, 1.0f - (xx + yy))};
}

// The game's network rotation quantisation (bin 384e30, 37c810 encode; 389010 decode): normalise, drop the largest
// component (ties to the lower index), make it positive, store the other three in 18 bits over +-1/sqrt(2), rebuild.
RLMAX_HD Q quantize_rot(Q in) {
    constexpr float M = 262143.0f;  // 2^18 - 1
    float q[4] = {in.x, in.y, in.z, in.w};
    const float n2 = q[1] * q[1] + q[0] * q[0] + q[2] * q[2] + q[3] * q[3];
    if (n2 <= 1e-8f) return Q{0.0f, 0.0f, 0.0f, 1.0f};
    const float inv = 1.0f / sqrtf(n2);
    for (float& v : q) v *= inv;
    const int a = absf(q[0]) < absf(q[1]) ? 1 : 0;
    const int b = absf(q[2]) < absf(q[3]) ? 3 : 2;
    const int drop = absf(q[b]) <= absf(q[a]) ? a : b;
    const float sign = q[drop] < 0.0f ? -1.0f : 1.0f;
    float st[3];
    for (int i = 0, o = 0; i < 4; i++) {
        if (i == drop) continue;
        const float f = (q[i] * sign * 1.4142135f * 0.5f + 0.5f) * M;
        int n = ((int)rintf(f + f + 0.5f)) >> 1;
        n = n < 0 ? 0 : (n > (int)M ? (int)M : n);
        const float t = (float)n / M - 0.5f;
        st[o++] = (t + t) * 0.70710677f;
    }
    const float d = sqrtf(((1.0f - st[0] * st[0]) - st[1] * st[1]) - st[2] * st[2]);
    float out[4];
    for (int i = 0, o = 0; i < 4; i++) out[i] = i == drop ? d : st[o++];
    const float m2 = out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3];
    if (m2 <= 1e-8f) return Q{0.0f, 0.0f, 0.0f, 1.0f};
    const float minv = 1.0f / sqrtf(m2);
    return Q{out[0] * minv, out[1] * minv, out[2] * minv, out[3] * minv};
}

RLMAX_HD void quantize_body(Body& b, V3 vel_uu) {
    // the rotation the game reads is Bullet's getRotation of the basis, not a cached quaternion: the rebuilt quaternion differs from the integrated one in the last bit, which
    // moves a tick-start codec tie by one 18-bit step (partycannon 9108: a car's basis 1e-5 off the game's)
    const Q q = quantize_rot(bullet_get_rotation(b.r));
    b.rot = q;
    b.r = bullet_set_rotation(q);
    // b.inv_i stays the tensor of the orientation the last integration produced: the game's rounding writes the rounded
    // transform without updateInertiaTensor (lead 57: the wheel friction's angular impulse matches the game's log to
    // the float noise on 26% of ticks with the stale tensor, 3% with a rebuilt one)
    V3 p = b.pos * c::BT_TO_UU;
    b.pos_uu = v3(q_pos(p.x, c::QUANT_POS_SCALE), q_pos(p.y, c::QUANT_POS_SCALE), q_pos(p.z, c::QUANT_POS_SCALE));
    b.pos = b.pos_uu * c::UU_TO_BT;
    const V3 v = vel_uu;
    b.vel_uu = v3(q_vel(v.x, c::QUANT_VEL_SCALE), q_vel(v.y, c::QUANT_VEL_SCALE), q_vel(v.z, c::QUANT_VEL_SCALE));
    b.vel = b.vel_uu * c::UU_TO_BT;
    b.ang = v3(q_vel(b.ang.x, c::QUANT_ANG_VEL_SCALE), q_vel(b.ang.y, c::QUANT_ANG_VEL_SCALE),
               q_vel(b.ang.z, c::QUANT_ANG_VEL_SCALE));
}

// Controls: asymmetric 8-bit axis quantisation. The vector form rounds ties away from zero, the scalar form to even
// (the difference is real on exact ties).
RLMAX_HD float quantize_axis_vec_elem(float x) {
    float cl = clampf(x, -1.0f, 1.0f);
    float scale = cl < 0.0f ? 128.0f : 127.0f;
    float biased = cl * scale + 128.0f;
    float w = biased + biased + 0.5f;
    int byte = (((int)round_away(w)) >> 1) & 0xFF;
    float s = (float)byte - 128.0f;
    return s < 0.0f ? s * (1.0f / 128.0f) : s / 127.0f;
}
RLMAX_HD V3 quantize_axis_inputs(V3 v) {
    return v3(quantize_axis_vec_elem(v.x), quantize_axis_vec_elem(v.y), quantize_axis_vec_elem(v.z));
}
RLMAX_HD float quantize_axis_input(float x) {
    float cl = clampf(x, -1.0f, 1.0f);
    float y = cl < 0.0f ? fmax_(cl * 128.0f, -128.0f) : fmin_(cl * 127.0f, 127.0f);
    float w = ((y + 128.0f) + (y + 128.0f)) + 0.5f;
    int eax = (int)round_even(w);
    int byte = (eax >> 1) & 0xFF;
    float s = (float)byte - 128.0f;
    return byte < 0x80 ? s * (1.0f / 128.0f) : s / 127.0f;
}

// ---- integration ----------------------------------------------------------------

// btTransformUtil::integrateTransform as the game runs it: the previous
// orientation is rebuilt from the basis (getRotation, scalar build), multiplied by the step quaternion in the scalar
// order, normalised by 1/sqrt and written back as a basis by setRotation (s = 2/|q|^2), not glam's from_quat.
RLMAX_HD void integrate_game(Body& b, V3 lin, V3 ang, float dt) {
    b.pos = b.pos + lin * dt;
    float angle = sqrtf((ang.x * ang.x + ang.y * ang.y) + ang.z * ang.z);
    if (angle * dt > c::ANGULAR_MOTION_THRESHOLD) angle = c::ANGULAR_MOTION_THRESHOLD / dt;
    // sin and cos of the half angle from one polynomial: the game's sin argument (angle * 0.5) * dt equals
    // (angle * dt) * 0.5 bit for bit wherever it is taken (angle >= 0.001: the halving is exact for a normal float)
    const float half_angle = (angle * dt) * 0.5f;
    const float sn = ucrt_sinf(half_angle), cs = ucrt_cosf(half_angle);
    float f;
    if (angle < 0.001f) {
        f = dt * 0.5f - ((((dt * dt) * dt) * 0.020833333333f) * angle) * angle;
    } else {
        f = sn / angle;
    }
    const float ax = ang.x * f, ay = ang.y * f, az = ang.z * f;
    const Q o = bullet_get_rotation(b.r);
    const float pw = ((cs * o.w - ax * o.x) - ay * o.y) - az * o.z;
    const float px = ((ax * o.w + cs * o.x) + ay * o.z) - az * o.y;
    const float py = ((cs * o.y + ay * o.w) + az * o.x) - ax * o.z;
    const float pz = ((cs * o.z + az * o.w) + ax * o.y) - ay * o.x;
    const float s = 1.0f / sqrtf(((px * px + py * py) + pz * pz) + pw * pw);
    const Q q{px * s, py * s, pz * s, pw * s};
    b.rot = q;
    b.r = bullet_set_rotation(q);
    // b.inv_i is not updated: the tensor of the new orientation is rebuilt by make_body at the next tick's start, and nothing
    // after the integration reads it (the solver reads it before; the post-tick, the pads and the state copy do not)
}

}  // namespace rlmax
