// RLMAX observation extract (the trainer's raw view of an arena): fixed-layout physical values in game units, no
// normalising, mirroring or frame change (the trainer does those). One row of floats per arena: the arena block, then
// one car block per car in car order. The field lists below are the single source: the GPU kernel (rlmax_extract_<NC>),
// the CPU build and the host's field-name table (cpu.cpp rlmax_extract_fields, cpu.rs extract_fields) all expand them.
//
// Arena block (EXTRACT_ARENA_F floats):
//   ball_pos_{x,y,z}      uu        ball_vel_{x,y,z}  uu/s       ball_ang_{x,y,z}  rad/s
//   tick                  the arena's tick counter (120 per second; kickoffs and goal resets do not restart it)
//   pad_0 .. pad_33       seconds until the boost pad is active again (0 = active), in pads.h order (y, then x)
// Car block (EXTRACT_CAR_F floats):
//   pos_{x,y,z} uu; vel_{x,y,z} uu/s; ang_{x,y,z} rad/s; fwd_{x,y,z}, up_{x,y,z} unit vectors (the car's x and z axes)
//   boost 0..100; on_ground, wheels (wheels in contact, 0..4), has_jumped, has_double_jumped, has_flipped, is_jumping,
//   is_flipping, is_auto_flipping, is_supersonic, is_boosting, is_demoed: 0 or 1 (wheels: a count)
//   jump_time, flip_time, air_time, time_since_jump, supersonic_time, demo_respawn_time, time_since_boosted: s
//   flip_torque_{x,y} the dodge direction (car frame); handbrake 0..1 (the ramped value)
//   team 0 blue / 1 orange; body preset (consts.h BodyPresetId)
//   ctl_{throttle,steer,pitch,yaw,roll} -1..1, ctl_{jump,boost,handbrake} 0 or 1: the controls of the last tick
#pragma once

#include "body.h"
#include "pads.h"
#include "state.h"

namespace rlmax {

#define RLMAX_EXTRACT_ARENA_FIELDS(X) \
    X(ball_pos_x) X(ball_pos_y) X(ball_pos_z) X(ball_vel_x) X(ball_vel_y) X(ball_vel_z) X(ball_ang_x) X(ball_ang_y) \
    X(ball_ang_z) X(tick)
#define RLMAX_EXTRACT_CAR_FIELDS(X)                                                                                      \
    X(pos_x) X(pos_y) X(pos_z) X(vel_x) X(vel_y) X(vel_z) X(ang_x) X(ang_y) X(ang_z) X(fwd_x) X(fwd_y) X(fwd_z) X(up_x)  \
    X(up_y) X(up_z) X(boost) X(on_ground) X(wheels) X(has_jumped) X(has_double_jumped) X(has_flipped) X(is_jumping)     \
    X(is_flipping) X(is_auto_flipping) X(is_supersonic) X(is_boosting) X(is_demoed) X(jump_time) X(flip_time)           \
    X(air_time) X(time_since_jump) X(supersonic_time) X(demo_respawn_time) X(time_since_boosted) X(flip_torque_x)  \
    X(flip_torque_y) X(handbrake) X(team) X(body) X(ctl_throttle) X(ctl_steer) X(ctl_pitch) X(ctl_yaw) X(ctl_roll)     \
    X(ctl_jump) X(ctl_boost) X(ctl_handbrake)

#define RLMAX_EXTRACT_ENUM(f) EX_##f,
enum ExtractArenaField : int { RLMAX_EXTRACT_ARENA_FIELDS(RLMAX_EXTRACT_ENUM) EX_ARENA_SCALARS };
enum ExtractCarField : int { RLMAX_EXTRACT_CAR_FIELDS(RLMAX_EXTRACT_ENUM) EXTRACT_CAR_F };
#undef RLMAX_EXTRACT_ENUM
constexpr int EX_PAD0 = EX_ARENA_SCALARS;
constexpr int EXTRACT_ARENA_F = EX_ARENA_SCALARS + NUM_PADS;

template <int NC>
RLMAX_HD constexpr int extract_row_floats() {
    return EXTRACT_ARENA_F + NC * EXTRACT_CAR_F;
}

// One arena's row. Written field by field into `out` (row-major, extract_row_floats<NC>() floats).
template <int NC>
RLMAX_HD void extract_arena(const ArenaState<NC>& s, float* out) {
    const BallState& b = s.ball;
    const V3 bp = b.pos * c::BT_TO_UU, bv = b.vel * c::BT_TO_UU;
    out[EX_ball_pos_x] = bp.x;
    out[EX_ball_pos_y] = bp.y;
    out[EX_ball_pos_z] = bp.z;
    out[EX_ball_vel_x] = bv.x;
    out[EX_ball_vel_y] = bv.y;
    out[EX_ball_vel_z] = bv.z;
    out[EX_ball_ang_x] = b.ang_vel.x;
    out[EX_ball_ang_y] = b.ang_vel.y;
    out[EX_ball_ang_z] = b.ang_vel.z;
    out[EX_tick] = (float)s.tick;
    for (int p = 0; p < NUM_PADS; p++) {
        const float cooldown = pad_def(p).big ? c::PAD_COOLDOWN_BIG : c::PAD_COOLDOWN_SMALL;
        const uint32_t gave = s.pad_gave_tick[p];
        out[EX_PAD0 + p] = gave == PAD_NONE ? 0.0f : fmax_(cooldown - (float)(int)(s.tick - gave) * c::DT, 0.0f);
    }
    for (int i = 0; i < NC; i++) {
        const CarState& cs = s.cars[i];
        float* o = out + EXTRACT_ARENA_F + i * EXTRACT_CAR_F;
        const V3 p = cs.pos * c::BT_TO_UU, v = cs.vel * c::BT_TO_UU;
        const M3 r = bullet_set_rotation(cs.rot);
        const uint32_t f = cs.flags;
        const auto bit = [f](uint32_t m) { return (f & m) ? 1.0f : 0.0f; };
        o[EX_pos_x] = p.x;
        o[EX_pos_y] = p.y;
        o[EX_pos_z] = p.z;
        o[EX_vel_x] = v.x;
        o[EX_vel_y] = v.y;
        o[EX_vel_z] = v.z;
        o[EX_ang_x] = cs.ang_vel.x;
        o[EX_ang_y] = cs.ang_vel.y;
        o[EX_ang_z] = cs.ang_vel.z;
        o[EX_fwd_x] = r.c0.x;
        o[EX_fwd_y] = r.c0.y;
        o[EX_fwd_z] = r.c0.z;
        o[EX_up_x] = r.c2.x;
        o[EX_up_y] = r.c2.y;
        o[EX_up_z] = r.c2.z;
        o[EX_boost] = cs.boost;
        o[EX_on_ground] = bit(CF_ON_GROUND);
        o[EX_wheels] = bit(CF_WHEEL0) + bit(CF_WHEEL0 << 1) + bit(CF_WHEEL0 << 2) + bit(CF_WHEEL0 << 3);
        o[EX_has_jumped] = bit(CF_HAS_JUMPED);
        o[EX_has_double_jumped] = bit(CF_HAS_DOUBLE_JUMPED);
        o[EX_has_flipped] = bit(CF_HAS_FLIPPED);
        o[EX_is_jumping] = bit(CF_IS_JUMPING);
        o[EX_is_flipping] = bit(CF_IS_FLIPPING);
        o[EX_is_auto_flipping] = bit(CF_IS_AUTO_FLIPPING);
        o[EX_is_supersonic] = bit(CF_IS_SUPERSONIC);
        o[EX_is_boosting] = bit(CF_IS_BOOSTING);
        o[EX_is_demoed] = bit(CF_IS_DEMOED);
        o[EX_jump_time] = (float)cs.jump_ticks * c::DT;
        o[EX_flip_time] = cs.flip_time;
        o[EX_air_time] = cs.air_time;
        o[EX_time_since_jump] = cs.time_since_jump;
        o[EX_supersonic_time] = cs.supersonic_timer;
        o[EX_demo_respawn_time] = cs.demo_respawn_timer;
        o[EX_time_since_boosted] = cs.time_since_boosted;
        o[EX_flip_torque_x] = cs.flip_torque_x;
        o[EX_flip_torque_y] = cs.flip_torque_y;
        o[EX_handbrake] = cs.handbrake_val;
        o[EX_team] = (float)cs.team;
        o[EX_body] = (float)cs.body;
        o[EX_ctl_throttle] = cs.ctl.throttle;
        o[EX_ctl_steer] = cs.ctl.steer;
        o[EX_ctl_pitch] = cs.ctl.pitch;
        o[EX_ctl_yaw] = cs.ctl.yaw;
        o[EX_ctl_roll] = cs.ctl.roll;
        o[EX_ctl_jump] = (cs.ctl.buttons & BTN_JUMP) ? 1.0f : 0.0f;
        o[EX_ctl_boost] = (cs.ctl.buttons & BTN_BOOST) ? 1.0f : 0.0f;
        o[EX_ctl_handbrake] = (cs.ctl.buttons & BTN_HANDBRAKE) ? 1.0f : 0.0f;
    }
}

}  // namespace rlmax
