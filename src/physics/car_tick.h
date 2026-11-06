// RLMAX car tick: the pre-tick (controls, drive, jump, flip, air control, auto-roll, boost, vehicle update) and the
// post-tick, in the game's order.
#pragma once

#include "car.h"

namespace rlmax {

// Tick-start rotation retained for contact callbacks.
struct CarSnap {
    Q q;  // the tick-start rotation
};

// What the drive, brake and steer of a tick read besides the forward speed: kept so that a car another car's wheels
// pushed can take them again at its new speed.
struct DriveIn {
    float raw_throttle, steer, handbrake, start_speed;
    int nwc;
    bool c_hb, boost_wants, undriven;
};

struct CarTickOut {
    // All four wheels share force and brake; front wheels share steer, rear wheels use zero.
    float engine_force, brake_force, steer;
    float real_throttle;  // the vehicle update's throttle input (it runs later inside the step, VEHICLE_IN_STEP)
    DriveIn drive;
    V3 f_auto, f_air, f_boost, f_jump;  // the parts of the tick's force, summed in this order after the sticky force
    bool has_auto, has_air, has_boost;
    uint32_t pushed;  // bit j: the wheels stuck car j's body to the ground; PUSHED_BALL: they stuck the ball's
};

constexpr uint32_t PUSHED_BALL = 1u << 31;

RLMAX_HD int wheel_count(uint32_t flags) {
    int n = 0;
    for (int i = 0; i < 4; i++) n += (int)((flags >> (16 + i)) & 1u);
    return n;
}
// last tick's wheels in contact with the world
RLMAX_HD int world_wheel_count(uint32_t flags) {
    int n = 0;
    for (int i = 0; i < 4; i++) n += (int)((flags >> (20 + i)) & 1u);
    return n;
}
RLMAX_HD bool fl(const CarState& c, uint32_t f) { return (c.flags & f) != 0; }
RLMAX_HD void set_fl(CarState& c, uint32_t f, bool v) { c.flags = v ? (c.flags | f) : (c.flags & ~f); }

RLMAX_HD V3 upwards_from_wheels(const CarState& cs, const Body& rb) {
    V3 s = cs.wheel_normal_sum;
    float l2 = len2(s);
    if (l2 > 0.0f && l2 < FLT_BIG) return s * (1.0f / sqrtf(l2));
    return rb.r.c2;
}

// Entry j (0..16383) of the game's sine table.
RLMAX_HD float ue3_table_sin(int j) {
    const float jf = (float)j;
    return ucrt_sinf(((jf + jf) * c::PI_F32) * c::UE3_TABLE_INV_SIZE);
}
RLMAX_HD float ue3_sin(int a) { return ue3_table_sin((a >> 2) & 16383); }
// An angle in radians as integer-rounded rotator units (before rounding)
RLMAX_HD float ue3_units(float radians) { return radians * c::UE3_HALF_TURN / c::PI_F32; }

// QuatToRotator(q) and its axes in integer UE3 units.
RLMAX_HD M3 ue3_rotator_axes(const Q& q) {
    const float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
    const float xx = q.x * x2, xy = q.x * y2, xz = q.x * z2, yy = q.y * y2, yz = q.y * z2, zz = q.z * z2;
    const float wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;
    const V3 fwd = v3(1.0f - (yy + zz), xy + wz, xz - wy), right = v3(xy - wz, 1.0f - (xx + zz), yz + wx), up = v3(xz + wy, yz - wx, 1.0f - (xx + yy));
    auto round = [](float v) { return ((int)rintf(v + v + 0.5f)) >> 1; };
    const int yaw = round(ue3_units(game_atan2f(fwd.y, fwd.x)));
    const int pitch = round(ue3_units(game_atan2f(fwd.z, sqrtf(fwd.y * fwd.y + fwd.x * fwd.x))));
    const float sy = ue3_sin(yaw), cy = ue3_sin(yaw + 16384);
    const int roll = round(ue3_units(game_atan2f((up.y * cy + up.x * -sy) + up.z * -0.0f, (right.y * cy + right.x * -sy) + right.z * -0.0f)));
    const float sp = ue3_sin(pitch), cp = ue3_sin(pitch + 16384), sr = ue3_sin(roll), cr = ue3_sin(roll + 16384);
    const float cr_sp = cr * sp, sr_sp = sr * sp;
    return M3{v3(cy * cp, cp * sy, sp), v3(sr_sp * cy - cr * sy, cy * cr + sr_sp * sy, -sr * cp),
              v3(-(cr_sp * cy + sy * sr), cy * sr - cr_sp * sy, cp * cr)};
}
RLMAX_HD V3 ue3_up_axis(const Q& q) { return ue3_rotator_axes(q).c2; }

template <int NC>
struct CarCtx {
    const ArenaData* ad;
    const CarGeo* g;  // per car
    Body* bodies;  // 0 ball, 1.. cars
    float ball_radius;
    const float (*box)[6];  // the last sweep's body boxes (ball, then cars), null when it took none: a wheel ray skips a body whose box misses its own
};

// One wheel's impulse as a velocity change, without touching the body. The world inverse inertia is read by rows
// (dot per row), as the game's SSE product does; in float it is not exactly symmetric.
struct WheelDv { V3 dv, dw; bool applied; };
RLMAX_HD WheelDv wheel_impulse(const Body& b, V3 impulse, V3 rel) {
    return WheelDv{impulse * b.inv_mass, mul_t(b.inv_i, cross(rel, impulse)), true};
}
RLMAX_HD void combine_wheels([[maybe_unused]] int ci, [[maybe_unused]] int src, Body& b, const WheelDv o[4]) {
    // the game visits its wheels 1, 0, 3, 2; its odd wheels negate y, ours even, so ours are 0, 1, 2, 3
    for (int i = 0; i < 4; i++) {
        if (!o[i].applied) continue;
        RLMAX_IMP_LAST(o[i].dv, o[i].dw, false);
        RLMAX_IMPW(ci, src, i, (b.vel += o[i].dv, b.ang += o[i].dw));
    }
}
// The steered wheel's axle: the car's right turned about its up by setRotation of the axis-angle quaternion.
RLMAX_HD V3 steered_axle(const M3& r, float angle) {
    const V3 up = r.c2;
    const float half = angle * 0.5f;
    const float sc = ucrt_sinf(half) / len(up);
    const M3 m = bullet_set_rotation(Q{up.x * sc, up.y * sc, up.z * sc, ucrt_cosf(half)});
    return m * r.c1;
}
// At zero angle, preserve the signed zeros in the quaternion matrix.
RLMAX_HD V3 zero_angle_axle(const M3& r, float half) {
    const V3 up = r.c2;
    const float x = up.x * half, y = up.y * half, z = up.z * half;
    const float xy = x * y, xz = x * z, yz = y * z;
    const M3 m{v3(1.0f, xy + z, xz - y), v3(xy - z, 1.0f, yz + x), v3(xz + y, yz - x, 1.0f)};
    return m * r.c1;
}
RLMAX_HD V3 steering_axle(const M3& r, float angle) {
    if (angle == 0.0f) {
        const float l2 = len2(r.c2);
        if (l2 > 0.0f && l2 <= FLT_BIG) return zero_angle_axle(r, angle * 0.5f);
    }
    return steered_axle(r, angle);
}
// btMatrix3x3::inverse of the world inverse inertia, as rows (c0 = row 0): the game turns a torque input into a torque with it.
RLMAX_HD M3 inertia_from_inverse(const M3& m) {
    const V3 r0 = m.c0, r1 = m.c1, r2 = m.c2;
    const float co_x = r1.y * r2.z - r1.z * r2.y;
    const float co_y = r1.z * r2.x - r1.x * r2.z;
    const float co_z = r1.x * r2.y - r1.y * r2.x;
    const float det = r0.x * co_x + r0.y * co_y + r0.z * co_z;
    const float s = det != 0.0f ? 1.0f / det : 0.0f;
    return M3{v3(co_x * s, (r0.z * r2.y - r0.y * r2.z) * s, (r0.y * r1.z - r0.z * r1.y) * s),
              v3(co_y * s, (r0.x * r2.z - r0.z * r2.x) * s, (r0.z * r1.x - r0.x * r1.z) * s),
              v3(co_z * s, (r0.y * r2.x - r0.x * r2.y) * s, (r0.x * r1.y - r0.y * r1.x) * s)};
}
RLMAX_HD bool torque_gate(V3 t) { return absf(t.x) >= 1.0e-4f || absf(t.y) >= 1.0e-4f || absf(t.z) >= 1.0e-4f; }
RLMAX_HD V3 ue_right(Q q) {
    const float z2 = q.z + q.z, y2 = q.y + q.y;
    const float x2x = (q.x + q.x) * q.x, x2w = (q.x + q.x) * q.w;
    return v3(y2 * q.x - z2 * q.w, 1.0f - (z2 * q.z + x2x), x2w + z2 * q.y);
}
// The game's UE quaternion axes and rotate.
RLMAX_HD V3 ue_up(Q q) {
    const float z2 = q.z + q.z, y2 = q.y + q.y;
    const float x2x = (q.x + q.x) * q.x, x2w = (q.x + q.x) * q.w;
    return v3(y2 * q.w + z2 * q.x, z2 * q.y - x2w, 1.0f - (y2 * q.y + x2x));
}
RLMAX_HD V3 ue_fwd_axis(Q q) {
    const float z2 = q.z + q.z, y2 = q.y + q.y;
    return v3(1.0f - (z2 * q.z + y2 * q.y), z2 * q.w + y2 * q.x, z2 * q.x - y2 * q.w);
}
// A force of `uu` (per unit mass, uu) along the car's up axis, as the impulse or force in BT that the jump applies.
RLMAX_HD V3 up_impulse(const Body& b, float uu) {
    const V3 u = ue_up(b.rot);
    const float scale = uu * b.mass;
    return v3((scale * u.x) * c::UU_TO_BT, (scale * u.y) * c::UU_TO_BT, (scale * u.z) * c::UU_TO_BT);
}
RLMAX_HD V3 ue_rotate_x(Q q) {
    const float qq = (q.y * q.y + q.x * q.x) + q.z * q.z;
    const float d = q.w * q.w - qq;
    const float w2 = q.w + q.w;
    const float t = q.x + q.x;
    return v3(d + t * q.x, (q.z * w2 + d * 0.0f) + t * q.y, (-q.y * w2 + d * 0.0f) + t * q.z);
}

// Psyclops jump and air torques.
RLMAX_HD V3 psyclops_torque(const Body& b, bool air) {
    const V3 p = b.pos * c::BT_TO_UU;
    float m = fabsf(q_pos(p.x, c::QUANT_POS_SCALE));
    const float y = fabsf(q_pos(p.y, c::QUANT_POS_SCALE));
    m = m <= y ? y : m;
    const float z = fabsf(q_pos(p.z, c::QUANT_POS_SCALE));
    m = m <= z ? z : m;
    const float pitch = ucrt_sinf(air ? m * 1.0e-4f : m);
    const float yaw = air ? ucrt_sinf(m * 2.0e-4f) : pitch;
    const float roll = air ? ucrt_sinf(m * 3.0e-4f) * 2.0f : pitch;
    return (ue_up(b.rot) * yaw + ue_right(b.rot) * pitch) + ue_fwd_axis(b.rot) * roll;
}
RLMAX_HD void psyclops_jump_torque(Body& b) {
    const V3 t = psyclops_torque(b, false);
    if (torque_gate(t)) b.ang += mul_t(b.inv_i, mul_t(inertia_from_inverse(b.inv_i), t));
}

template <int NC, bool Psyclops = false>
RLMAX_HD void vehicle_update(int ci, CarState& cs, const CarCtx<NC>& x, float handbrake_val, float real_throttle,
                             CarTickOut& out) {
    Body& rb = x.bodies[1 + ci];
    const CarGeo& g = x.g[ci];
    out.pushed = 0;
    V3 cpos = rb.pos;
    M3 cr = rb.r;
    float friction_scale = rb.mass / 3.0f;
    // Only the current vehicle needs its wheel working set. The pre-tick retains three shared inputs per car.
    Wheel wheels[4];
    for (int i = 0; i < 4; ++i) {
        wheels[i].engine_force = out.engine_force;
        wheels[i].brake = out.brake_force;
        wheels[i].steer_angle = i < 2 ? out.steer : 0.0f;
    }
    int cell = cell_of(*x.ad, cpos + cr * g.hitbox_off);
    // all four rays first, against the world as it is before any wheel's impulses
    WheelHit hits[4];
    V3 ray_from[4], ray_to[4];
    for (int i = 0; i < 4; i++) {
        Wheel& w = wheels[i];
        w.hard_point = cpos + cr * g.wheel_conn[i];
        ray_from[i] = w.hard_point;
        ray_to[i] = w.hard_point - cr.c2 * g.ray_len[i];
    }
    wheel_rays4<NC>(*x.ad, ray_from, ray_to, ci, x.bodies, x.g, x.ball_radius, x.box, cell, hits);
    int num_contact = 0;
    // An axle is read only by wheels that hit. Airborne axles need no steering trig or matrix work.
    const V3 front_axle = (hits[0].ok || hits[1].ok) ? steering_axle(cr, wheels[0].steer_angle) : v3s(0.0f);
    const V3 back_axle = (hits[2].ok || hits[3].ok) ? steering_axle(cr, 0.0f) : v3s(0.0f);
    for (int i = 0; i < 4; i++) {
        Wheel& w = wheels[i];
        w.hit = hits[i].ok;
        if (!w.hit) {
            w.extra_pushback = 0.0f;
            w.in_contact_world = false;
            continue;
        }
        num_contact++;
        bool front = i < 2;
        const V3 axle = front ? front_axle : back_axle;
        // apply_ray_cast
        int gb = hits[i].body;
        bool world = gb < 0;
        w.in_contact_world = world;
        w.ground_body = gb;
        w.contact_point = hits[i].point;
        w.contact_normal = hits[i].normal;
        w.axle_dir = axle;
        V3 up = cr.c2;
        float trace = dot(w.hard_point - w.contact_point, up);
        float max_len = g.rest_len[i] + c::MAX_SUSPENSION_TRAVEL_UU * c::UU_TO_BT;
        w.vel_at_contact = vel_at(rb, w.contact_point - cpos);
        float proj = dot(w.contact_normal, w.vel_at_contact);
        float denom = dot(w.contact_normal, up);
        if (denom > 0.1f) {
            float inv = 1.0f / denom;
            w.susp_rel_vel = proj * inv;
            w.clipped_inv = inv;
        } else {
            w.susp_rel_vel = 0.0f;
            w.clipped_inv = 10.0f;
        }
        w.susp_len = fmin_(trace - g.radius[i], max_len);
        w.extra_pushback = 0.0f;
        bool car_hit = gb >= 1;
        if (world || car_hit) {
            float thresh = (g.rest_len[i] + g.radius[i]) - 2.5f * c::UU_TO_BT;
            if (trace < thresh) {
                const Body* other = car_hit ? &x.bodies[gb] : nullptr;
                float res = resolve_single_collision(rb, other, w.contact_point, w.contact_normal, trace - thresh);
                w.extra_pushback = res / 4.0f;
            }
        }
        // refresh_friction_curves
        V3 long_dir = cross(axle, w.contact_normal);
        V3 wheel_delta = w.hard_point - cpos;
        V3 cv = (cross(rb.ang, wheel_delta) + rb.vel) * c::BT_TO_UU;
        float base = absf(dot(cv, axle));
        float fin = base > 5.0f ? base / (absf(dot(cv, long_dir)) + base) : 0.0f;
        float lat_f = Psyclops ? curves::PSYCLOPS_LAT_FRICTION.eval(fin) : curves::LAT_FRICTION.eval(fin);
        float long_f = 1.0f;
        if (handbrake_val != 0.0f) {
            lat_f *= 1.0f + (curves::HANDBRAKE_LAT_FRICTION_FACTOR.eval(fin) - 1.0f) * handbrake_val;
            long_f *= 1.0f + (curves::HANDBRAKE_LONG_FRICTION_FACTOR.eval(fin) - 1.0f) * handbrake_val;
        }
        if (real_throttle == 0.0f || !world) {
            float ns = curves::NON_STICKY_FRICTION_FACTOR.eval(w.contact_normal.z);
            lat_f *= ns;
            long_f *= ns;
        }
        w.lat_friction = lat_f;
        w.long_friction = long_f;
        // apply_single_wheel_dynamic_effects: on a non-world body the victim pushback, then the ground stick (the order of two float additions)
        if (!world) {
            Body& ground = x.bodies[gb];
            out.pushed |= gb > 0 ? 1u << (gb - 1) : PUSHED_BALL;
            if (w.extra_pushback > 0.0f && car_hit) {
                add_rel(ground, -w.contact_normal * (w.extra_pushback * 4.0f), w.contact_point - ground.pos, true, false);
            }
            add_rel(ground, -w.contact_normal, w.contact_point - ground.pos, true, false);
        }
    }
    if (num_contact < 3) {
        for (int i = 0; i < 4; i++) wheels[i].engine_force /= 4.0f;
    }
    for (int i = 0; i < 4; i++) RLMAX_WHEEL_DBG(ci, i, wheels[i]);
    // No wheel contributes to either phase when none touched.
    if (num_contact != 0) {
        // update_suspension: every wheel's impulse from the same state, then combined
        WheelDv wd[4];
        for (int i = 0; i < 4; i++) {
            wd[i].applied = false;
            Wheel& w = wheels[i];
            if (!w.hit) continue;
            float force = (g.rest_len[i] - w.susp_len) * c::SUSPENSION_STIFFNESS * w.clipped_inv;
            float damp = w.susp_rel_vel < 0.0f ? c::WHEELS_DAMPING_COMPRESSION : c::WHEELS_DAMPING_RELAXATION;
            float sf = (force - damp * w.susp_rel_vel) * g.force_scale[i];
            if (sf <= 0.0f) continue;
            float bscale = sf * c::DT + w.extra_pushback;
            wd[i] = wheel_impulse(rb, w.contact_normal * bscale, w.contact_point - cpos);
        }
        combine_wheels(ci, IL_SUSP, rb, wd);
        // update_friction_impulse and apply_friction_impulses: again from one state, after the suspension
        for (int i = 0; i < 4; i++) {
            wd[i].applied = false;
            Wheel& w = wheels[i];
            if (!w.hit) continue;
            const V3 axle = w.axle_dir;
            const V3 cn = cross(w.contact_normal, axle);
            const V3 fwd = cn * (1.0f / sqrtf(len2(cn)));
            float rolling = 0.0f;
            if (w.engine_force == 0.0f) {
                if (w.brake != 0.0f) {
                    const float rel = dot(w.vel_at_contact, fwd);
                    rolling = clampf(-rel * c::ROLLING_FRICTION_SCALE, -w.brake, w.brake) * (c::DT * friction_scale);
                }
            } else {
                rolling = -w.engine_force * c::DT;
            }
            const float side_imp = resolve_bilateral_fake_ground(rb, w.contact_point, axle);
            const float roll = rolling * w.long_friction;
            const float side = (side_imp * (c::DT * friction_scale)) * w.lat_friction;
            w.impulse = v3(roll * fwd.x + side * axle.x, roll * fwd.y + side * axle.y, roll * fwd.z + side * axle.z);
            if (len2(w.impulse) == 0.0f) continue;
            V3 off = w.contact_point - cpos;
            V3 rel = off - cr.c2 * dot(cr.c2, off);
            if constexpr (Psyclops) rel -= cr.c2 * (4.0f * c::UU_TO_BT);
            wd[i] = wheel_impulse(rb, w.impulse, rel);
        }
        combine_wheels(ci, IL_FRIC, rb, wd);
    }
    // update_contact_state, the sticky gate and the normals the next tick's sticky force reads
    uint32_t bits = 0;
    bool any_world = false;
    V3 nsum = v3s(0.0f);
    for (int i = 0; i < 4; i++) {
        if (wheels[i].hit) bits |= 1u << (16 + i);
        if (wheels[i].hit && wheels[i].in_contact_world) {
            bits |= 1u << (20 + i);
            any_world = true;
            nsum += wheels[i].contact_normal;
        }
    }
    cs.flags = (cs.flags & ~(0xFFu << 16)) | bits;
    set_fl(cs, CF_ON_GROUND, wheel_count(cs.flags) >= 3);
    set_fl(cs, CF_STICKY_GATE, any_world);
    cs.wheel_normal_sum = nsum;
}

// Vehicle_TA::GetForwardSpeed and the drive/steer callback's speed (bin efeae0, f72740): the vehicle's quaternion applied to (1,0,0) by
// 374760, dotted with the linear velocity in uu/s in the order y, x, z. All in float, in the decompile's order.
// Forward speed from a fresh snapshot: the body's rotation and velocity put through the game's codec again.
RLMAX_HD float fresh_forward_speed(const Body& b, bool before_step = false) {
    const Q q = quantize_rot(bullet_get_rotation(b.r));
    V3 bt = b.vel;
    // the vehicle reads the snapshot after Bullet clamps the body's speed in BT units;
    // the pre-tick runs earlier here, so that clamp is taken here without changing the body
    const float max_bt = c::CAR_MAX_SPEED_UU * c::UU_TO_BT;
    if (before_step && len2(bt) > max_bt * max_bt) bt = normalize_or_zero(bt) * max_bt;
    V3 v = bt * c::BT_TO_UU;
    const float l2 = (v.y * v.y + v.x * v.x) + v.z * v.z;
    if (c::CAR_MAX_SPEED_UU * c::CAR_MAX_SPEED_UU < l2) {
        const float inv = 1.0f / sqrtf(l2);
        v = v3(c::CAR_MAX_SPEED_UU * (inv * v.x), c::CAR_MAX_SPEED_UU * (inv * v.y), c::CAR_MAX_SPEED_UU * (inv * v.z));
    }
    const V3 u = v3(q_vel(v.x, c::QUANT_VEL_SCALE), q_vel(v.y, c::QUANT_VEL_SCALE), q_vel(v.z, c::QUANT_VEL_SCALE));
    const float w2 = q.w + q.w;
    const float s = q.w * q.w - ((q.y * q.y + q.x * q.x) + q.z * q.z);
    const float d = q.x + q.x;
    const float fx = s + d * q.x;
    const float fy = (q.z * w2 + s * 0.0f) + d * q.y;
    const float fz = (-q.y * w2 + s * 0.0f) + d * q.z;
    return (fx * u.x + fy * u.y) + fz * u.z;
}
RLMAX_HD float game_forward_speed(const Body& b) {
    const Q q = b.rot;
    const float w2 = q.w + q.w;
    const float s = q.w * q.w - (q.y * q.y + q.x * q.x + q.z * q.z);
    const float d = q.x + 0.0f + 0.0f;
    const float t = d + d;
    const float fx = (0.0f * q.y - 0.0f * q.z) * w2 + s * 1.0f + t * q.x;
    const float fy = (1.0f * q.z - 0.0f * q.x) * w2 + s * 0.0f + t * q.y;
    const float fz = (0.0f * q.x - 1.0f * q.y) * w2 + s * 0.0f + t * q.z;
    const V3 v = b.vel_uu;
    return (fx * v.x + fy * v.y) + fz * v.z;
}

// Drive, brake, steer and the sticky force at a given forward speed; returns the sticky force. It runs again after a
// dodge or double-jump impulse (car_pre_tick), and for a car pushed by an earlier car's wheels (car_redrive).
template <int NC, bool Psyclops = false>
RLMAX_HD V3 drive_wheels(const DriveIn& d, int ci, const CarState& cs, const Body& rb, const CarCtx<NC>& x, float fwd,
                         CarTickOut& out) {
    V3 sticky = v3s(0.0f);
    float real_brake = 0.0f;
    V3 pts = quantize_axis_inputs(v3(d.raw_throttle, d.steer, 0.0f));
    const float proc_throttle = pts.x, proc_steer = pts.y;
    const float rthr = (d.boost_wants && d.nwc == 4) ? 1.0f : proc_throttle;
    const float abs_fwd = absf(fwd);
    float engine_throttle = d.undriven ? 0.0f : rthr;
    if (!d.c_hb && !d.undriven) {
        if (absf(rthr) > c::THROTTLE_DEADZONE) {
            if (abs_fwd > c::STOPPING_FORWARD_VEL_UU && signum(rthr) != signum(fwd)) {
                real_brake = 1.0f;
                if (abs_fwd > c::BRAKING_NO_THROTTLE_SPEED_THRESH) engine_throttle = 0.0f;
            }
        } else {
            engine_throttle = 0.0f;
            real_brake = abs_fwd < c::STOPPING_FORWARD_VEL_UU ? 1.0f : c::COASTING_BRAKE_FACTOR;
        }
    }
    const float drive_scale = curves::DRIVE_SPEED_TORQUE_FACTOR.eval(fwd);
    const float engine_force = ((drive_scale * (engine_throttle * c::DRIVE_TORQUE_UU)) * c::WHEEL_TORQUE_SHARE) * c::UU_TO_BT;
    const float brake_force = ((real_brake * c::BRAKE_TORQUE_UU) * c::WHEEL_TORQUE_SHARE) * c::UU_TO_BT;
    // steering (bin f72740): the init-averaged curve at |forward speed|, blended toward the powerslide curve by the
    // handbrake factor, times the steer input; one angle for both front wheels
    float steer_deg;
    if constexpr (Psyclops) {
        constexpr Curve<2> steer{{0.0f, 2300.0f}, {19.622270584106445f, 1.9959354400634766f}};
        steer_deg = steer.eval(abs_fwd);
    } else {
        steer_deg = x.g[ci].steer.eval(abs_fwd);
    }
    if (d.handbrake != 0.0f) steer_deg += (x.g[ci].steer_powerslide.eval(abs_fwd) - steer_deg) * d.handbrake;
    const float steer = steer_deg * proc_steer * c::DEG_TO_RAD;
    out.engine_force = engine_force;
    out.brake_force = brake_force;
    out.steer = steer;
    const bool sticky_gate = quirk::STICKY_GATE_PREV_TICK ? fl(cs, CF_STICKY_GATE) : true;
    if (sticky_gate) {
        V3 updir = upwards_from_wheels(cs, rb);
        bool full_stick = d.raw_throttle != 0.0f || d.start_speed > c::STOPPING_FORWARD_VEL_UU;
        // the three-wheel sticky force has zero base.
        const float base = Psyclops ? 0.0f : 0.5f;
        const float st = full_stick ? 1.5f : base;
        const float sfs = absf(updir.z) * (base - st) + st;
        sticky = ((updir * (c::GRAVITY_Z_UU * sfs)) * c::UU_TO_BT) * rb.mass;
    }
    return sticky;
}

template <int NC, bool Psyclops = false>
RLMAX_HD void car_pre_tick(int ci, CarState& cs, const CarCtx<NC>& x, CarTickOut& out) {
    Body& rb = x.bodies[1 + ci];
    const V3 pre_ang = rb.ang;
    Controls& ctl = cs.ctl;
    ctl.throttle = clampf(ctl.throttle, -1.0f, 1.0f);
    ctl.steer = clampf(ctl.steer, -1.0f, 1.0f);
    ctl.pitch = clampf(ctl.pitch, -1.0f, 1.0f);
    ctl.yaw = clampf(ctl.yaw, -1.0f, 1.0f);
    ctl.roll = clampf(ctl.roll, -1.0f, 1.0f);
    const bool c_jump = (ctl.buttons & BTN_JUMP) != 0;
    const bool c_boost = (ctl.buttons & BTN_BOOST) != 0;
    const bool c_hb = (ctl.buttons & BTN_HANDBRAKE) != 0;
    const float fwd_speed = game_forward_speed(rb);
    const bool jump_pressed = c_jump && !fl(cs, CF_PREV_JUMP);
    const int nwc = wheel_count(cs.flags);
    const bool on_ground = fl(cs, CF_ON_GROUND);

    // ---- update_wheels
    cs.handbrake_val =
        clampf(cs.handbrake_val + (c_hb ? c::POWERSLIDE_RISE_RATE : -c::POWERSLIDE_FALL_RATE) * c::DT, 0.0f, 1.0f);
    const float raw_throttle = ctl.throttle;
    // drive, brake, steer and the sticky force at a given forward speed; returns the sticky velocity delta. It runs
    // again after a dodge or double-jump impulse (below)
    const float start_speed = absf(fwd_speed);
    out.drive = DriveIn{raw_throttle, ctl.steer, cs.handbrake_val, start_speed, nwc, c_hb, c_boost && cs.boost > 0.0f, (ctl.buttons & BTN_UNDRIVEN) != 0};
    V3 sticky_force = drive_wheels<NC, Psyclops>(out.drive, ci, cs, rb, x, fresh_forward_speed(rb, quirk::VEHICLE_IN_STEP), out);
    V3 jump_force = v3s(0.0f);  // summed last, as the game queues it
    bool jump_started = false;

    // a landing that cancels a flip skips auto-roll too (cw_double_jump_dash, cw_double_dash)
    const bool flipping_at_start = fl(cs, CF_IS_FLIPPING);
    if (on_ground) {
        if (fl(cs, CF_IS_FLIPPING)) cs.flip_time = 0.0f;
        set_fl(cs, CF_IS_FLIPPING, false);
    }

    // the jump re-arms as three wheels land, or on the ground once no jump has run for 0.125 s
    {
        const bool ground = nwc >= 3;
        if (ground && (!fl(cs, CF_JUMP_GROUND) || (fl(cs, CF_HAS_JUMPED) && !fl(cs, CF_IS_JUMPING) && cs.time_since_jump >= c::JUMP_REARM_TIME))) {
            set_fl(cs, CF_HAS_JUMPED, false);
            set_fl(cs, CF_AIR_JUMP_USED, false);
        }
        set_fl(cs, CF_JUMP_GROUND, ground);
    }
    // ---- update_jump
    const bool was_jumping = fl(cs, CF_IS_JUMPING);
    {
        if (!fl(cs, CF_HAS_JUMPED) && on_ground && jump_pressed) {
            set_fl(cs, CF_IS_JUMPING, true);
            set_fl(cs, CF_HAS_JUMPED, true);
            cs.jump_ticks = 0;
            set_fl(cs, CF_JUMP_HOLD_BROKEN, false);
        }
        cs.jump_ticks += 1;
        if (fl(cs, CF_IS_JUMPING)) {
            if (!c_jump) set_fl(cs, CF_JUMP_HOLD_BROKEN, true);
            bool still = cs.jump_ticks <= c::JUMP_MIN_TICKS ||
                         (!fl(cs, CF_JUMP_HOLD_BROKEN) && c_jump && cs.jump_ticks <= c::JUMP_MAX_TICKS);
            set_fl(cs, CF_IS_JUMPING, still);
            if (!still) cs.jump_ticks = 1;
        }
        if (fl(cs, CF_IS_JUMPING)) {
            if (cs.jump_ticks == 1) {
                jump_started = true;  // the game starts a jump before the wheels; they read the new velocity
                if constexpr (Psyclops) psyclops_jump_torque(rb);
                RLMAX_IMP(ci, IL_JUMP, add_lin(rb, up_impulse(rb, c::JUMP_IMMEDIATE_FORCE_UU), true, false));
            }
            jump_force = up_impulse(rb, c::JUMP_ACCEL_UU);
        }
    }

    // ---- update_auto_flip: the car lies on its roof by its rotator's up axis
    if (jump_pressed && fl(cs, CF_WORLD_CONTACT) && cs.world_contact_normal.z > c::AUTOFLIP_NORM_Z_THRESH &&
        dot(cs.world_contact_normal, ue3_up_axis(rb.rot)) < c::AUTOFLIP_UP_DOT) {
        const float roll = -game_atan2f(rb.r.c1.z, rb.r.c2.z);
        const float flip_time = Psyclops ? c::AUTOFLIP_TIME_THREE_WHEELS : c::AUTOFLIP_TIME;
        cs.auto_flip_timer = (float)((double)flip_time * ((double)absf(roll) / 3.14159265358979323846));
        cs.auto_flip_torque_scale = roll > 0.0f ? 1.0f : -1.0f;
        set_fl(cs, CF_IS_AUTO_FLIPPING, true);
        RLMAX_IMP(ci, IL_AUTOFLIP, add_lin(rb, ((-ue_up(rb.rot) * c::AUTOFLIP_IMPULSE_UU) * c::UU_TO_BT) * rb.mass, true, false));
    }
    if (fl(cs, CF_IS_AUTO_FLIPPING)) {
        if (cs.auto_flip_timer <= 0.0f) {
            set_fl(cs, CF_IS_AUTO_FLIPPING, false);
            cs.auto_flip_timer = 0.0f;
        } else {
            const V3 t = (ue_fwd_axis(rb.rot) * c::AUTOFLIP_ANG_VEL_DELTA) * cs.auto_flip_torque_scale;
            rb.ang += mul_t(rb.inv_i, mul_t(inertia_from_inverse(rb.inv_i), t));
            cs.auto_flip_timer -= c::DT;
        }
    }

    // ---- update_double_jump_or_flip
    const bool prev_is_flipping = fl(cs, CF_IS_FLIPPING);
    const float prev_flip_time = cs.flip_time;
    bool flip_ended = false;
    bool redrive = false;
    // the air's rules follow the wheels, not the on-ground flag: on a tick after a state was set the two differ
    if (nwc >= 3) {
        set_fl(cs, CF_HAS_DOUBLE_JUMPED, false);
        set_fl(cs, CF_HAS_FLIPPED, false);
        set_fl(cs, CF_AIR_JUMP_USED, false);
        cs.air_time = 0.0f;
        cs.flip_time += c::DT;
    } else {
        cs.air_time += c::DT;
        // the air jump's window closes only for a car that jumped, long enough after the jump
        if (jump_pressed && !fl(cs, CF_AIR_JUMP_USED) && !(fl(cs, CF_HAS_JUMPED) && cs.time_since_jump > c::DOUBLEJUMP_MAX_DELAY)) {
            V3 di = quantize_axis_inputs(v3(-ctl.pitch, ctl.yaw, ctl.roll));
            di.x = quantize_axis_input(-ctl.pitch);
            float mag = absf(di.y) + absf(di.x) + absf(di.z);
            bool is_flip = mag >= 0.5f;  // dodge deadzone (CarBodyConfig default 0.5)
            bool can_use = !fl(cs, CF_IS_AUTO_FLIPPING) && !fl(cs, CF_HAS_DOUBLE_JUMPED) && !fl(cs, CF_HAS_FLIPPED);
            if (can_use) {
                set_fl(cs, CF_HAS_JUMPED, true);
                if (is_flip) {
                    cs.flip_time = 0.0f;
                    set_fl(cs, CF_HAS_FLIPPED, true);
                    set_fl(cs, CF_AIR_JUMP_USED, true);
                    set_fl(cs, CF_IS_FLIPPING, true);
                    // the impulse through the mass; the car's integer UE3 yaw from the game's atan2f and the
                    // rotator's rounding, indexing the 16384-entry sine table in steps of 4 units
                    const float side_in = di.y + di.z;
                    V3 dd = (absf(side_in) < 0.1f && absf(di.x) < 0.1f) ? v3s(0.0f) : normalize_or_zero(v3(di.x, side_in, 0.0f));
                    cs.flip_torque_x = -dd.y;
                    cs.flip_torque_y = dd.x;
                    if (absf(dd.x) < 0.1f) dd.x = 0.0f;
                    if (absf(dd.y) < 0.1f) dd.y = 0.0f;
                    if (len2(dd) > 0.0f) {
                        const float abs_speed = absf(fwd_speed);
                        const float ratio = abs_speed / c::CAR_MAX_SPEED_UU;
                        const bool back = abs_speed < 100.0f ? dd.x < 0.0f : ((dd.x >= 0.0f) != (fwd_speed >= 0.0f));
                        const float imp = c::FLIP_INITIAL_VEL_SCALE * rb.mass;
                        const float imp_x = back ? rintf(imp * c::FLIP_BACKWARD_IMPULSE_SCALE_X) : imp;
                        float ix = dd.x * imp_x, iy = dd.y * imp;
                        const float msx = back ? c::FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE : c::FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE;
                        ix *= ((msx - 1.0f) * ratio) + 1.0f;
                        iy *= ((c::FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE - 1.0f) * ratio) + 1.0f;
                        const Q q = rb.rot;
                        const float y2 = q.y + q.y, z2 = q.z + q.z;
                        const float fwd_y = q.x * y2 + q.w * z2, fwd_x = 1.0f - (q.y * y2 + q.z * z2);
                        const float yv = ue3_units(game_atan2f(fwd_y, fwd_x));
                        const int yaw = ((int)rintf(yv + yv + 0.5f)) >> 1;
                        const int yi = (yaw >> 2) & 16383;
                        const V3 f2 = v3(ue3_table_sin((yi + 4096) & 16383), ue3_table_sin(yi), 0.0f);
                        const V3 r2 = v3(-f2.y, f2.x, 0.0f);
                        const V3 dv = f2 * ix + r2 * iy;
                        RLMAX_IMP(ci, IL_DODGE, add_lin(rb, dv * c::UU_TO_BT, true, false));
                        redrive = true;
                    }
                } else {
                    if constexpr (Psyclops) psyclops_jump_torque(rb);
                    RLMAX_IMP(ci, IL_DJUMP, add_lin(rb, up_impulse(rb, c::JUMP_IMMEDIATE_FORCE_UU), true, false));
                    redrive = true;
                    set_fl(cs, CF_HAS_DOUBLE_JUMPED, true);
                    set_fl(cs, CF_AIR_JUMP_USED, true);
                }
            }
        }
        if (fl(cs, CF_IS_FLIPPING)) {
            float ft = cs.flip_time;
            bool still = fl(cs, CF_HAS_FLIPPED) && ft < c::FLIP_TORQUE_TIME;
            set_fl(cs, CF_IS_FLIPPING, still);
            cs.flip_time = still ? ft + c::DT : c::DT;
            flip_ended = !still;
        } else {
            cs.flip_time += c::DT;
        }
    }
    // a dodge or double-jump impulse this tick: drive, brake, steer and the sticky force read the post-impulse forward
    // speed (run after the flip-before-wheels step; RLBot c3 car1 f96764 landing dodge
    // matches to 0.01 uu/s; RLMAX fit S03 dodge_landing_0 k108 recorded vy 495.62, 470.83 at the tick-start speed)
    if (redrive || jump_started) sticky_force = drive_wheels<NC, Psyclops>(out.drive, ci, cs, rb, x, fresh_forward_speed(rb, quirk::VEHICLE_IN_STEP), out);

    // ---- update_air_torque
    const V3 ue_f = ue_fwd_axis(rb.rot), ue_r = ue_right(rb.rot), ue_u = ue_up(rb.rot);
    V3 dodge_torque = v3s(0.0f);
    bool has_dodge_torque = false;
    V3 torque_items[3], air_force = v3s(0.0f);  // the queue's torques in order: the Psyclops boost, the air input, the air damping
    int torque_n = 0;
    bool has_air_force = false;
    if constexpr (Psyclops) {
        if (nwc == 0 && !fl(cs, CF_WORLD_CONTACT) && fl(cs, CF_IS_BOOSTING)) {
            const V3 t = psyclops_torque(rb, true);
            if (torque_gate(t)) torque_items[torque_n++] = mul_t(inertia_from_inverse(rb.inv_i), t);
        }
    }
    if (nwc < 3) {
        const bool allow_air = nwc == 0;
        if (fl(cs, CF_IS_FLIPPING) && (cs.flip_torque_x != 0.0f || cs.flip_torque_y != 0.0f)) {
            V3 rel = v3(cs.flip_torque_x, cs.flip_torque_y, 0.0f);
            float pscale = 1.0f;
            if (rel.y != 0.0f && ctl.pitch != 0.0f && signum(rel.y) == signum(ctl.pitch) &&
                prev_flip_time >= c::FLIP_PITCH_CANCEL_GATE_MIN_TIME) {
                pscale = 1.0f - fmin_(absf(quantize_axis_input(-ctl.pitch)), 1.0f);
            }
            // each component rounded to whole torque units before the pitch-cancel scale (the recorded dodges match it to 1e-4 rad/s, RLMAX's unrounded form drifts)
            V3 tq = v3(round_even(rel.x * c::FLIP_TORQUE_X), round_even(rel.y * c::FLIP_TORQUE_Y), 0.0f);
            tq.y *= pscale;
            const V3 tw = v3(((tq.y * ue_r.x + tq.x * ue_f.x) + tq.z * ue_u.x) + 0.0f,
                             ((tq.y * ue_r.y + tq.x * ue_f.y) + tq.z * ue_u.y) + 0.0f,
                             ((tq.y * ue_r.z + tq.x * ue_f.z) + tq.z * ue_u.z) + 0.0f);
            if (torque_gate(tw)) { dodge_torque = mul_t(inertia_from_inverse(rb.inv_i), tw); has_dodge_torque = true; }
        }
        // The three-wheel car cannot air-control during a world contact.
        if (allow_air && (!Psyclops || !fl(cs, CF_WORLD_CONTACT))) {
            V3 in = quantize_axis_inputs(v3(ctl.pitch, ctl.yaw, ctl.roll));
            bool any = !is_zero(in);
            bool lock = any && (prev_is_flipping || fl(cs, CF_IS_FLIPPING) || prev_flip_time < c::FLIP_PITCHLOCK_EXTRA_TIME);
            float pitch_scale = lock ? 0.0f : 1.0f;
            const float in_pitch = in.x * pitch_scale;
            const float a_p = (c::AIR_TORQUE_P * -in_pitch) * c::UE3_UNIT_TO_RAD;
            const float a_y = (c::AIR_TORQUE_Y * in.y) * c::UE3_UNIT_TO_RAD;
            const float a_r = (-c::AIR_TORQUE_R * in.z) * c::UE3_UNIT_TO_RAD;
            const V3 t_in = v3((((a_p * ue_r.x + 0.0f) + a_y * ue_u.x) + a_r * ue_f.x) * 1000.0f,
                               (((a_p * ue_r.y + 0.0f) + a_y * ue_u.y) + a_r * ue_f.y) * 1000.0f,
                               (((a_p * ue_r.z + 0.0f) + a_y * ue_u.z) + a_r * ue_f.z) * 1000.0f);
            const float d_p = (1.0f - absf(in_pitch)) * (dot(pre_ang, ue_r) * (c::AIR_DAMP_P * c::UE3_UNIT_TO_RAD));
            const float d_y = (1.0f - absf(in.y)) * (dot(pre_ang, ue_u) * (c::AIR_DAMP_Y * c::UE3_UNIT_TO_RAD));
            const float d_r = (1.0f * dot(pre_ang, ue_f)) * (c::AIR_DAMP_R * c::UE3_UNIT_TO_RAD);
            const V3 t_damp = v3((((0.0f - d_p * ue_r.x) - d_y * ue_u.x) - d_r * ue_f.x) * 1000.0f,
                                 (((0.0f - d_p * ue_r.y) - d_y * ue_u.y) - d_r * ue_f.y) * 1000.0f,
                                 (((0.0f - d_p * ue_r.z) - d_y * ue_u.z) - d_r * ue_f.z) * 1000.0f);
            const bool apply_in = torque_gate(t_in), apply_damp = torque_gate(t_damp);
            if (apply_in || apply_damp) {
                const M3 iw = inertia_from_inverse(rb.inv_i);
                if (apply_in) torque_items[torque_n++] = mul_t(iw, t_in);
                if (apply_damp) torque_items[torque_n++] = mul_t(iw, t_damp);
            }
            float ts = fl(cs, CF_IS_BOOSTING) ? 1.0f : quantize_axis_input(ctl.throttle);
            if (ts != 0.0f) { air_force = ((ue_f * (c::THROTTLE_AIR_ACCEL_UU * rb.mass)) * ts) * c::UU_TO_BT; has_air_force = true; }
        }
    }

    // ---- auto-roll (skipped on the tick a flip ends)
    V3 auto_force = v3s(0.0f), auto_torque = v3s(0.0f);
    bool has_auto = false;
    const float quant_throttle = quantize_axis_input(ctl.throttle);
    const bool world_contact = fl(cs, CF_WORLD_CONTACT);
    // auto-roll counts last tick's world-contact wheels only
    const int nww = world_wheel_count(cs.flags);
    // the three-wheel path does not apply auto-roll.
    if (!Psyclops && quant_throttle != 0.0f && !fl(cs, CF_IS_FLIPPING) && !flipping_at_start && !flip_ended && nww < 4 && (nww > 0 || world_contact)) {
        const bool gate = fl(cs, CF_STICKY_GATE);
        if (gate || world_contact) {
            V3 gup = (gate && nww > 0) ? upwards_from_wheels(cs, rb) : cs.world_contact_normal;
            V3 gdown = -gup;
            V3 f = rb.r.c0, r = rb.r.c1;
            V3 cr = cross(gup, f);
            V3 cf = cross(gdown, cr);
            float rtf = 1.0f - clampf(dot(r, cr), 0.0f, 1.0f);
            float ftf = 1.0f - clampf(dot(f, cf), 0.0f, 1.0f);
            V3 tdr = f * -signum(dot(r, gup));
            V3 tdf = r * signum(dot(f, gup));
            auto_force = gdown * c::AUTOROLL_FORCE_UU * c::UU_TO_BT * rb.mass;
            const V3 G = gup;
            const float crx = ue_f.z * G.y - ue_f.y * G.z, cry = ue_f.x * G.z - ue_f.z * G.x, crz = ue_f.y * G.x - ue_f.x * G.y;
            const float r_dot = clampf((crx * ue_r.x + ue_r.y * cry) + ue_r.z * crz, 0.0f, 1.0f);
            const float nx = -G.x, ny = -G.y, nz = -G.z;
            const float f_dot = clampf(((ny * crz - nz * cry) * ue_f.x + (crx * nz - nx * crz) * ue_f.y) + (nx * cry - crx * ny) * ue_f.z, 0.0f, 1.0f);
            const V3 f_axis = 0.0f <= (ue_r.y * G.y + ue_r.x * G.x) + ue_r.z * G.z ? -ue_f : ue_f;
            const V3 r_axis = (ue_f.y * G.y + ue_f.x * G.x) + ue_f.z * G.z < 0.0f ? -ue_r : ue_r;
            const float rf = (1.0f - r_dot) * c::AUTOROLL_TORQUE, ff = (1.0f - f_dot) * c::AUTOROLL_TORQUE;
            const V3 t_game = v3(ff * r_axis.x + rf * f_axis.x, ff * r_axis.y + rf * f_axis.y, ff * r_axis.z + rf * f_axis.z);
            auto_torque = mul_t(inertia_from_inverse(rb.inv_i), t_game);
            has_auto = true;
        }
    }
    set_fl(cs, CF_WORLD_CONTACT, false);

    // the tick's sums in the game's order: auto-roll first when it applies, the dodge torque after the air control
    if (has_auto) out.f_auto = auto_force;
    if (has_air_force) out.f_air = air_force;
    out.has_auto = has_auto;
    out.has_air = has_air_force;
    out.has_boost = false;
    rb.force = has_auto ? auto_force + sticky_force : sticky_force;
    if (has_air_force) rb.force += air_force;
    // the auto-roll torque leads the queue and the others follow one by one
    rb.torque = has_auto ? auto_torque : v3s(0.0f);
    for (int q = 0; q < torque_n; q++) rb.torque += torque_items[q];
    if (has_dodge_torque) rb.torque += dodge_torque;

    // the time since the car last jumped, on the ground too
    cs.time_since_jump = fl(cs, CF_IS_JUMPING) ? 0.0f : cs.time_since_jump + c::DT;

    // ---- update_boost
    {
        const V3 bdir = ue_rotate_x(rb.rot);
        float accel = on_ground ? c::BOOST_ACCEL_GROUND_UU : c::BOOST_ACCEL_AIR_UU;
        if (fl(cs, CF_IS_BOOSTING)) {
            cs.boost = fmax_(cs.boost - c::BOOST_USED_PER_SECOND * c::DT, 0.0f);
            bool depleted = cs.boost == 0.0f;
            bool latch_expired = !c_boost && cs.boosting_time >= c::BOOST_MIN_TIME;
            if (depleted || latch_expired) {
                set_fl(cs, CF_IS_BOOSTING, false);
                cs.boosting_time = 0.0f;
                cs.time_since_boosted += c::DT;
            } else {
                cs.boosting_time += c::DT;
                cs.time_since_boosted = 0.0f;
                out.f_boost = (bdir * (accel * rb.mass)) * c::UU_TO_BT;
                out.has_boost = true;
                rb.force += out.f_boost;
            }
        } else if (c_boost && cs.boost > 0.0f) {
            set_fl(cs, CF_IS_BOOSTING, true);
            cs.boosting_time += c::DT;
            cs.time_since_boosted = 0.0f;
            out.f_boost = (bdir * (accel * rb.mass)) * c::UU_TO_BT;
            out.has_boost = true;
            rb.force += out.f_boost;
        } else {
            set_fl(cs, CF_IS_BOOSTING, false);
            cs.boosting_time = 0.0f;
            cs.time_since_boosted += c::DT;
        }
        cs.boost = clampf(cs.boost, 0.0f, c::BOOST_MAX);
        if (fl(cs, CF_IS_JUMPING)) out.f_jump = jump_force;
        rb.force += jump_force;
    }

    // boost drives the wheels only with all four down, as the drive's own throttle: a boosting car on fewer has no sticky friction
    const float real_throttle = (out.drive.boost_wants && nwc == 4) ? 1.0f : quant_throttle;
    out.real_throttle = real_throttle;
    if (!quirk::VEHICLE_IN_STEP) vehicle_update<NC, Psyclops>(ci, cs, x, cs.handbrake_val, real_throttle, out);
}

// A car that an earlier car's wheels pushed, taken after its own pre-tick: the game runs each car's pre-tick and wheels
// in turn, so its drive, brake, steer and sticky force read the pushed velocity (the
// vehicle update of each car follows the previous one's). The force is summed again in the same order, and gravity
// is added to the pre-gravity acceleration, which is positive zero in the vehicle-in-step path.
template <int NC, bool HasPsyclops = true>
RLMAX_HD void car_redrive(int ci, CarState& cs, const CarCtx<NC>& x, CarTickOut& out) {
    Body& rb = x.bodies[1 + ci];
    V3 sticky;
    if constexpr (HasPsyclops)
        sticky = cs.body == BODY_PSYCLOPS ? drive_wheels<NC, true>(out.drive, ci, cs, rb, x, fresh_forward_speed(rb), out)
                                         : drive_wheels<NC>(out.drive, ci, cs, rb, x, fresh_forward_speed(rb), out);
    else sticky = drive_wheels<NC>(out.drive, ci, cs, rb, x, fresh_forward_speed(rb), out);
    rb.force = out.has_auto ? out.f_auto + sticky : sticky;
    if (out.has_air) rb.force += out.f_air;
    if (out.has_boost) rb.force += out.f_boost;
    // The jump state cannot change before redrive; absent force was positive zero.
    rb.force += fl(cs, CF_IS_JUMPING) ? out.f_jump : v3s(0.0f);
    const V3 total = rb.force + v3(0.0f, 0.0f, (c::GRAVITY_Z_UU * c::UU_TO_BT) * rb.mass);
    rb.acc_lin = v3s(0.0f) + (total * rb.inv_mass) * c::DT;
}

RLMAX_HD void car_post_tick(CarState& cs, const Body& rb, const CarTickOut& out) {
    const float START2 = c::SUPERSONIC_START_SPEED_UU * c::SUPERSONIC_START_SPEED_UU;
    const float MAINT2 = c::SUPERSONIC_MAINTAIN_MIN_SPEED_UU * c::SUPERSONIC_MAINTAIN_MIN_SPEED_UU;
    float sp2 = len2(rb.vel * c::BT_TO_UU);
    if (fl(cs, CF_IS_SUPERSONIC)) {
        if (sp2 >= START2) {
            cs.supersonic_timer = 0.0f;
        } else if (sp2 < MAINT2) {
            set_fl(cs, CF_IS_SUPERSONIC, false);
            cs.supersonic_timer = 0.0f;
        } else {
            cs.supersonic_timer += c::DT;
            if (cs.supersonic_timer >= c::SUPERSONIC_MAINTAIN_MAX_TIME) {
                set_fl(cs, CF_IS_SUPERSONIC, false);
                cs.supersonic_timer = 0.0f;
            }
        }
    } else if (sp2 >= START2) {
        set_fl(cs, CF_IS_SUPERSONIC, true);
        cs.supersonic_timer = 0.0f;
    } else {
        cs.supersonic_timer = 0.0f;
    }
    const bool on_ground = fl(cs, CF_ON_GROUND);
    cs.bump_cooldown = fmax_(cs.bump_cooldown - c::DT, 0.0f);
    set_fl(cs, CF_PREV_JUMP, (cs.ctl.buttons & (BTN_JUMP | BTN_JUMP_HELD)) != 0);
}

}  // namespace rlmax
