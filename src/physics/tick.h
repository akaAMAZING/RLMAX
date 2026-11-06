// RLMAX tick: one physics tick of one arena, in the game's order. This is the CPU composition of the stages; the GPU build runs the same
// stage functions as separate kernels.
#pragma once

#include "car_tick.h"
#include "demo.h"
#include "manifold.h"
#include "pads.h"
#include "solver.h"
#include "sweep.h"

namespace rlmax {

// Stage timers for the profiling build only (RLMAX_PROFILE): lane 0 of each warp adds the warp's cycles per stage, so
// divergence shows as time. The exact and fast builds compile them away.
#if defined(__CUDA_ARCH__) && defined(RLMAX_PROFILE)
__device__ unsigned long long g_prof[32];
#define RLMAX_PROF_BEGIN long long _pt = clock64();
#define RLMAX_PROF(k)                                                                        \
    {                                                                                        \
        long long _t = clock64();                                                            \
        if ((threadIdx.x & 31) == 0) atomicAdd(&g_prof[k], (unsigned long long)(_t - _pt)); \
        _pt = _t;                                                                            \
    }
#else
#define RLMAX_PROF_BEGIN
#define RLMAX_PROF(k)
#endif

// (1 - 0.03)^(1/120) in f32, as the game computes it for the ball each tick.
constexpr float BALL_DAMPING_PER_TICK = 0.9997462034225464f;

// BALL_UPLOADED: a host-written state (set_ball); the tick start decides whether it is asleep: exactly still and below the awake rest height.
enum BallActivation : uint32_t { BALL_ACTIVE = 0, BALL_WANTS_DEACTIVATION = 1, BALL_SLEEPING = 2, BALL_UPLOADED = 3 };

constexpr int BALL_WORLD_SLOTS = BALL_WORLD_MANIFOLDS * MANIFOLD_CAPACITY;
constexpr int CAR_WORLD_SLOTS = CAR_WORLD_MANIFOLDS * MANIFOLD_CAPACITY;
constexpr int CAR_CAR_SLOTS = 4;
constexpr int CAR_BALL_SLOTS = 4;

// The tick's solver: a contact row per contact slot plus the ball-world synthetic row; a friction row per non-special
// slot plus one. Named here so the local-memory whitelist (rlmax_local_sizes) measures the type the tick uses.
template <int NC>
constexpr int tick_rows() {
    return NC * CAR_WORLD_SLOTS + NC * CAR_BALL_SLOTS + NC * (NC - 1) / 2 * CAR_CAR_SLOTS + BALL_WORLD_SLOTS + 1 + NC + 1;
}
template <int NC>
constexpr int tick_friction_rows() {
    return NC * CAR_WORLD_SLOTS + NC * CAR_BALL_SLOTS + NC * (NC - 1) / 2 * CAR_CAR_SLOTS + 1 + NC + 1;
}
template <int NC>
using TickSolver = Solver<tick_rows<NC>(), tick_friction_rows<NC>(), 1 + NC>;

struct TickConsts {
    float ball_radius, ball_threshold;
    V3 ball_inv_inertia;
    V3 gravity;
};

RLMAX_HD TickConsts tick_consts() {
    TickConsts k;
    k.ball_radius = c::BALL_RADIUS_BT;
    k.ball_threshold = (k.ball_radius + c::SPHERE_BOUND_PAD_BT) * c::CONTACT_BREAKING_THRESHOLD;
    float bi = 0.4f * c::BALL_MASS_BT * k.ball_radius * k.ball_radius;
    k.ball_inv_inertia = v3s(1.0f / bi);
    k.gravity = v3(0.0f, 0.0f, c::GRAVITY_Z_UU * c::UU_TO_BT);
    return k;
}

RLMAX_HD Body make_body(V3 pos, Q rot, V3 vel, V3 ang, float mass, V3 inv_i_local) {
    Body b;
    b.pos = pos;
    b.vel = vel;
    b.ang = ang;
    b.acc_lin = b.acc_ang = b.force = b.torque = v3s(0.0f);
    b.vel_uu = vel * c::BT_TO_UU;
    b.pos_uu = pos * c::BT_TO_UU;
    b.mass = mass;
    b.inv_mass = 1.0f / mass;
    b.inv_i_local = inv_i_local;
    body_set_rot(b, rot);
    b.active = true;
    b.contact_response = true;
    return b;
}

// Arena reset spots (sim/consts.rs car::spawn)
RLMAX_HD void respawn_spot(int i, float* x, float* y) {
    if (i == 0) { *x = -2304.f; *y = -4608.f; }
    else if (i == 1) { *x = -2688.f; *y = -4608.f; }
    else if (i == 2) { *x = 2304.f; *y = -4608.f; }
    else { *x = 2688.f; *y = -4608.f; }
}

// The rotation a respawned car starts with: through the game's integer rotator and its sine table (yaw 16384), orange
// turned by the yaw 32768 quaternion, whose sine is sin(pi as a float)
RLMAX_HD Q respawn_rot(uint32_t team) {
    return team == 0 ? Q{-0.0f, -0.0f, 0x1.6a09e8p-1f, 0x1.6a09e4p-1f} : Q{0.0f, 0.0f, 0x1.6a09e2p-1f, -0x1.6a09eap-1f};
}

// The basis its first tick's tensor comes from: the rotator's yaw alone, blue 16384, orange -16384, taken through the
// quaternion's axes and back through Bullet
RLMAX_HD M3 respawn_inertia_basis(uint32_t team) {
    const M3 axes = team == 0 ? M3{v3(-0x1p-23f, 0x1.fffffep-1f, 0.0f), v3(-0x1.fffffep-1f, -0x1p-23f, -0.0f), v3(-0.0f, 0.0f, 1.0f)}
                              : M3{v3(0x1p-24f, -0x1.fffffep-1f, 0.0f), v3(0x1.fffffep-1f, 0x1p-24f, 0.0f), v3(0.0f, 0.0f, 1.0f)};
    return bullet_set_rotation(bullet_get_rotation(axes));
}

RLMAX_HD Q yaw_quat(float yaw) { return q_axis_angle(v3(0, 0, 1), yaw); }

RLMAX_HD uint32_t rng_next(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}

template <int NC>
RLMAX_HD void respawn_car(CarState& cs, uint32_t& rng) {
    float x, y;
    respawn_spot((int)(rng_next(rng) % 4u), &x, &y);
    float ydir = cs.team == 0 ? -1.0f : 1.0f;  // Team::get_y_dir: blue -1, orange +1
    uint32_t team = cs.team;
    CarState n = {};
    n.team = team;
    n.body = cs.body;  // a car keeps its hitbox through a demolition
    n.ctl = cs.ctl;  // and this tick's controls: it drives at once
    // on the ground by flag, though no wheel touches yet; the game's jump state outlives the car
    n.flags = (cs.flags & (CF_HAS_JUMPED | CF_AIR_JUMP_USED | CF_JUMP_GROUND)) | CF_ON_GROUND;
    n.time_since_jump = cs.time_since_jump;
    n.pad_touch = PAD_NONE;
    n.pos = v3(x, y * -ydir, respawn_z_uu(cs.body)) * c::UU_TO_BT;
    n.rot = respawn_rot(team);
    n.boost = c::BOOST_SPAWN_AMOUNT;
    n.wman.n = 0;
    cs = n;
}

// Kickoff spawns: five blue spots, x and y in uu and
// yaw; orange takes the point mirror (-x, -y, yaw + pi). Boost 33.3, cars at rest.
RLMAX_HD void kickoff_spot(int i, float* x, float* y, float* yaw) {
    if (i == 0) { *x = -2048.f; *y = -2560.f; *yaw = PI * 0.25f; }
    else if (i == 1) { *x = 2048.f; *y = -2560.f; *yaw = PI * 0.75f; }
    else if (i == 2) { *x = -256.f; *y = -3840.f; *yaw = PI * 0.5f; }
    else if (i == 3) { *x = 256.f; *y = -3840.f; *yaw = PI * 0.5f; }
    else { *x = 0.f; *y = -4608.f; *yaw = PI * 0.5f; }
}

#ifdef RLMAX_FULL_STATE
// Test-only: the respawn spot a recorded state carries for a demolished car (full_state.h g_fs_respawn), in place of the rng's pick
template <int NC>
RLMAX_HD void fs_respawn_override(CarState& cs, int i) {
    const uint32_t ov = g_fs_respawn[i];
    if (g_fs_spawn_z[i] > 0.0f) cs.pos.z = g_fs_spawn_z[i] * c::UU_TO_BT;
    if (!ov) return;
    g_fs_respawn[i] = 0;
    float x, y, yaw;
    const uint32_t team = cs.team;
    const uint32_t body = cs.body;
    CarState n = {};
    n.team = team;
    n.body = body;
    n.ctl = cs.ctl;
    n.flags = (cs.flags & (CF_HAS_JUMPED | CF_AIR_JUMP_USED | CF_JUMP_GROUND)) | CF_ON_GROUND;
    n.time_since_jump = cs.time_since_jump;
    n.pad_touch = PAD_NONE;
    n.boost = c::BOOST_SPAWN_AMOUNT;
    if (ov <= 4) {
        respawn_spot((int)ov - 1, &x, &y);
        const float ydir = team == 0 ? -1.0f : 1.0f;
        n.pos = v3(x, y * -ydir, g_fs_spawn_z[i] > 0.0f ? g_fs_spawn_z[i] : respawn_z_uu(body)) * c::UU_TO_BT;
        n.rot = respawn_rot(team);
    } else {
        kickoff_spot((int)ov - 5, &x, &y, &yaw);
        if (team) { x = -x; y = -y; yaw += PI; }
        const KickoffRest& rest = kickoff_rest(body);
        const Q turn = yaw_quat(yaw - PI * 0.5f);
        n.pos = v3(x * c::UU_TO_BT, y * c::UU_TO_BT, rest.z);
        n.rot = turn * Q{rest.rot[0], rest.rot[1], rest.rot[2], rest.rot[3]};
        n.vel = q_rotate(turn, v3(rest.vel[0], rest.vel[1], rest.vel[2]));
        n.ang_vel = q_rotate(turn, v3(rest.ang[0], rest.ang[1], rest.ang[2]));
        n.flags = CF_ON_GROUND | CF_JUMP_GROUND | CF_STICKY_GATE | (0xFu << 16) | (0xFu << 20);
        n.wheel_normal_sum = v3(0.0f, 0.0f, 4.0f);
        n.jump_ticks = 480;
        n.flip_time = 480.0f * c::DT;
    }
    n.wman.n = 0;
    cs = n;
}
#endif

// Reset an arena to a kickoff: the five spots shuffled from the arena's rng (blue car k on spot k, orange car k on its
// mirror; which arrangement the game picks is open), the ball at rest on the centre spot, contact caches and pads
// cleared, events cleared.
template <int NC>
RLMAX_HD void kickoff_reset(ArenaState<NC>& s) {
    s.seed_mask = 0;
    int perm[5] = {0, 1, 2, 3, 4};
    for (int k = 4; k > 0; k--) {
        const int j = (int)(rng_next(s.rng) % (uint32_t)(k + 1));
        const int t = perm[k];
        perm[k] = perm[j];
        perm[j] = t;
    }
    BallState& b = s.ball;
    // at rest at z 92.75, 0.4 uu inside the 93.15 rest height (every recorded kickoff: exactly still): no gravity until a
    // car touches it. It sleeps at the end of the first tick, after that tick builds its ball-world manifold: the game's
    // ball has lain on the floor through the countdown, and a car's first touch wakes it only with those contacts (starting
    // asleep and empty, first touch vz was off by 5 to 69 uu/s on our own captures; exact with this)
    b.pos = v3(0, 0, c::BALL_KICKOFF_Z_UU * c::UU_TO_BT);
    b.rot = q_identity();
    b.vel = b.ang_vel = v3s(0.0f);
    b.deactivation_time = 0.0f;
    b.sleeping = BALL_WANTS_DEACTIVATION;
    b.sleep_inv_i = world_inv_inertia(bullet_set_rotation(b.rot), tick_consts().ball_inv_inertia);
    int team_count[2] = {0, 0};
    for (int i = 0; i < NC; i++) {
        CarState& cs = s.cars[i];
        uint32_t team = (uint32_t)(i % 2);
        int spot = i / 2;
        if constexpr (NC == 3 || NC == 5 || NC == 8) {
            team = cs.team == 0 ? 0u : 1u;
            spot = team_count[team]++;
        }
        const uint32_t body = cs.body;
        float x, y, yaw;
        kickoff_spot(perm[spot % 5], &x, &y, &yaw);
        if (team) {
            x = -x;
            y = -y;
            yaw += PI;
        }
        // at rest as the game's cars are at "go" (consts.h kickoff_rest): the body's settled height, tilt, wheel contacts and
        // sticky gate, turned about z from the table's yaw pi/2 to this spot's; x and y are the spot's
        const KickoffRest& rest = kickoff_rest(body);
        const Q turn = yaw_quat(yaw - PI * 0.5f);
        CarState n = {};
        n.team = team;
        n.body = body;
        n.pos = v3(x * c::UU_TO_BT, y * c::UU_TO_BT, rest.z);
        n.rot = turn * Q{rest.rot[0], rest.rot[1], rest.rot[2], rest.rot[3]};
        n.vel = q_rotate(turn, v3(rest.vel[0], rest.vel[1], rest.vel[2]));
        n.ang_vel = q_rotate(turn, v3(rest.ang[0], rest.ang[1], rest.ang[2]));
        n.boost = c::BOOST_SPAWN_AMOUNT;
        // the four wheels in contact with the world, as at rest (the settled flags word 0x2ff0c00)
        n.flags = CF_ON_GROUND | CF_JUMP_GROUND | CF_STICKY_GATE | (0xFu << 16) | (0xFu << 20);
        n.wheel_normal_sum = v3(0.0f, 0.0f, 4.0f);
        // the game's counters at "go" (the recordings: jump_time and flip_time 3.99 to 4.00 s): flip_time under 0.3 s locks
        // the air-control pitch (car_tick.h update_air_torque), so a counter at 0 would lock a jump in the first 0.3 s
        n.jump_ticks = 480;
        n.flip_time = 480.0f * c::DT;
        n.pad_touch = PAD_NONE;
        cs = n;
    }
    for (int p = 0; p < (NC * (NC - 1) / 2 > 0 ? NC * (NC - 1) / 2 : 1); p++) s.pair_man[p].n = 0;
    for (int i = 0; i < NC; i++) s.ball_man[i].n = 0;
    s.sweep.hdr = 0;  // the broadphase sweep is rebuilt from the new poses (sweep.h)
    sweep_round_reset<NC>(s.sweep);
    s.ball_wman.n = 0;
    for (int p = 0; p < NUM_PADS; p++) s.pad_gave_tick[p] = PAD_NONE;
    s.events = 0;
}

// Recover a tick-start car velocity for contact callbacks. The car state keeps
// the unquantized velocity until post-tick; respawn velocities are below the cap.
RLMAX_COLD V3 contact_car_velocity_uu(const CarState& car) {
    static_assert(quirk::QUANTIZE_PHYS, "callback velocity uses tick-start quantization");
    V3 v = car.vel * c::BT_TO_UU;
    if (len2(v) > c::CAR_MAX_SPEED_UU * c::CAR_MAX_SPEED_UU)
        v = normalize_or_zero(v) * c::CAR_MAX_SPEED_UU;
    return v3(q_vel(v.x, c::QUANT_VEL_SCALE), q_vel(v.y, c::QUANT_VEL_SCALE), q_vel(v.z, c::QUANT_VEL_SCALE));
}

// The same tick-start angular clamp and codec, needed only for a demolition cast.
RLMAX_COLD V3 contact_car_angular(const CarState& car) {
    static_assert(quirk::QUANTIZE_PHYS, "callback angular velocity uses tick-start quantization");
    V3 v = car.ang_vel;
    if (len2(v) > c::CAR_MAX_ANG_SPEED * c::CAR_MAX_ANG_SPEED)
        v = normalize_or_zero(v) * c::CAR_MAX_ANG_SPEED;
    return v3(q_vel(v.x, c::QUANT_ANG_VEL_SCALE), q_vel(v.y, c::QUANT_ANG_VEL_SCALE), q_vel(v.z, c::QUANT_ANG_VEL_SCALE));
}

// The game's UE3 rotator sine: the angle in 65536ths of a turn, the 16384-entry table in steps of 4.
// The car-car contact callbacks (bump and demolition), after the step on the tick-start states; out of line on the
// GPU: rare, and the cones and the time-of-impact bisection would otherwise add register pressure to the tick.
template <int NC>
RLMAX_COLD void car_car_callbacks(ArenaState<NC>& s, const CarSnap* snap, const CarGeo* geo,
                                  const ContactPoint (*car_car)[CAR_CAR_SLOTS], V3* bumped_vel, EpaScratch& scratch) {
    // a car demolished this tick by its victim may still demolish it but not bump it; a car
    // demolished earlier, or by another car, takes no part
    int demoer_now[NC];
    for (int i = 0; i < NC; i++) demoer_now[i] = -1;
    int p = 0;
    for (int i = 0; i < NC; i++) {
        for (int j = i + 1; j < NC; j++, p++) {
            for (int q = 0; q < (int)s.pair_man[p].n; q++) {  // the pair's held points: the slots written below
                const ContactPoint& cp = car_car[p][q];
                for (int sw = 0; sw < 2; sw++) {
                    int ai = sw ? j : i, vi = sw ? i : j;
                    CarState& att = s.cars[ai];
                    CarState& vic = s.cars[vi];
                    if (fl(vic, CF_IS_DEMOED)) continue;
                    const bool demo_only = fl(att, CF_IS_DEMOED);
                    if (demo_only && demoer_now[ai] != vi) continue;
                    V3 apos = att.pos * c::BT_TO_UU, vpos = vic.pos * c::BT_TO_UU;
                    V3 avel = contact_car_velocity_uu(att), vvel = contact_car_velocity_uu(vic);
                    V3 delta = vpos - apos;
                    if (dot(avel, delta) < 0.0f) continue;
                    V3 vel_dir = normalize_or_zero(avel);
                    V3 dir_to_v = normalize_or_zero(delta);
                    if (dot(avel, dir_to_v) <= dot(vvel, vel_dir)) continue;
                    // Tick-start quantization rebuilds the basis from this quaternion.
                    static_assert(quirk::QUANTIZE_PHYS, "callback basis follows the quantized quaternion");
                    const CarPose ap{att.pos, bullet_set_rotation(snap[ai].q), snap[ai].q, apos},
                                  vp{vic.pos, bullet_set_rotation(snap[vi].q), snap[vi].q, vpos};
                    const BodyPreset& ac = body_preset(att.body), &vc = body_preset(vic.body);
                    const V3 aoff = v3(ac.off_x, 0.0f, ac.off_z), voff = v3(vc.off_x, 0.0f, vc.off_z);
                    // demolition is checked before the bump cooldown, at the poses of first contact
                    bool is_demo = false;
                    if (fl(att, CF_IS_SUPERSONIC)) {
                        const float fs = dot(avel, ue3_rotator_axes(snap[ai].q).c0);
                        if (absf(fs) >= c::SUPERSONIC_MAINTAIN_MIN_SPEED_UU) {
                            CarPose at = ap, vt = vp;
                            time_of_impact_poses(at, geo[ai], avel, contact_car_angular(att), vt, geo[vi], vvel, contact_car_angular(vic), scratch);
                            is_demo = within_forward_cone(at, aoff, vt, voff, fs < 0.0f, c::DEMO_YAW_DEG, c::DEMO_PITCH_DEG);
                        }
                    }
                    if (is_demo) is_demo = att.team != vic.team;
                    if (is_demo) {
                        set_fl(vic, CF_IS_DEMOED, true);
                        vic.demo_respawn_timer = c::DEMO_RESPAWN_TIME;
                        demoer_now[vi] = ai;
                        continue;
                    }
                    if (demo_only) continue;
                    if (att.bump_cooldown > 0.0f && (!fl(att, CF_HAS_BUMP_VICTIM) || att.bump_victim == (uint32_t)vi)) continue;
                    const bool bumper = within_forward_cone(ap, aoff, vp, voff, false, c::BUMP_YAW_DEG, c::BUMP_PITCH_DEG);
                    if (bumper) {
                        // the impulse in momentum units along the attacker's velocity (its reciprocal speed), then
                        // through 1/50 and 1/mass; on the ground the victim's up is its rotator's
                        const float speed = sqrtf(len2(avel));
                        const V3 dir = avel * (1.0f / speed);
                        auto curve = [speed](float y0, float y1, float y2) {
                            if (speed < 1400.0f) return y0 + (y1 - y0) * (speed / 1400.0f);
                            if (speed < 2200.0f) return y1 + (y2 - y1) * ((speed - 1400.0f) / 800.0f);
                            return y2;
                        };
                        const V3 impulse = fl(vic, CF_ON_GROUND) ? dir * curve(150.0f, 200000.0f, 275000.0f) + ue3_up_axis(snap[vi].q) * curve(50.0f, 50000.0f, 75000.0f)
                                                                 : dir * curve(150.0f, 250000.0f, 350000.0f);
                        bumped_vel[vi] += (impulse * (1.0f / 50.0f)) * (1.0f / c::CAR_MASS_BT);
                    }
                    // the next bump needs the physics time strictly past the cooldown (half a tick longer); the
                    // cooldown starts without a bump too
                    att.bump_cooldown = c::BUMP_COOLDOWN_TIME + c::DT * 0.5f;
                    att.bump_victim = (uint32_t)vi;
                    set_fl(att, CF_HAS_BUMP_VICTIM, true);
                }
            }
        }
    }
}

// An uploaded ball is asleep when exactly still and below the awake rest height: the kickoff ball sits 0.4 uu inside it,
// which a simulated ball cannot hold; a state-set ball resting at 93.15 or hanging in the air is awake (tolerance 0.2 uu). Out of line on the GPU: taken once after each upload.
RLMAX_COLD void resolve_uploaded_ball(BallState& bs) {
    bs.sleeping = (is_zero(bs.vel) && is_zero(bs.ang_vel) && bs.pos.z * c::BT_TO_UU < c::BALL_REST_Z_UU - 0.2f)
                      ? BALL_SLEEPING : BALL_ACTIVE;
}

// A ball uploaded asleep has lain where it is: the game's resting ball keeps the floor or wall point its last awake tick made,
// and a car that wakes it is solved against that point. Prime each empty plane slot with the point the detection makes at
// the ball's pose (a sleeping ball's pose is its previous pose). Shared by the large and the staged tick.
RLMAX_COLD void prime_uploaded_ball_planes(WorldManifolds<BALL_WORLD_MANIFOLDS>& w, const Body& ball, float radius, float thr) {
    for (int p = 0; p < NUM_PLANES + 2; p++) {
        if (wman_holds(w, 0x100u + (uint32_t)p)) continue;
        const PlaneXf px = plane_xf(p);
        const Contact cp = sphere_plane(ball.pos, ball.r, radius, px, thr);
        if (!cp.ok || cp.d > thr) continue;
        Manifold* m = wman_get(w, 0x100u + (uint32_t)p);
        if (m) man_add_point(*m, MPoint{man_to_local(&ball, cp.pb + cp.n * cp.d), cp.pb - px.o, cp.n, cp.d}, thr);
    }
}

// The arena planes (bit p: plane p, sweep handle 17 + p) whose manifold the ball holds. A sleeping ball makes no new static
// manifold in the dispatcher's array but keeps one it holds, in its place ahead of the car manifolds made after it.
RLMAX_HD uint32_t ball_held_planes(const WorldManifolds<BALL_WORLD_MANIFOLDS>& w) {
    uint32_t held = 0;
    for (uint32_t q = 0; q < w.n; q++)
        if (w.obj[q] >= 0x100u && w.obj[q] < 0x104u && w.m[q].n) held |= 1u << (w.obj[q] - 0x100u);
    return held;
}

// Bullet predictive point cast.
struct PredictiveHit { V3 n; float d; };
RLMAX_COLD bool predictive_world(const ArenaData& ad, const Body& body, const CarGeo* geo, PredictiveHit& cp) {
    // This central box is disjoint from every mesh AABB and all arena planes. Bound both child centres and the ray backoff.
    const float reach = geo ? absf(geo->hitbox_off.x) + absf(geo->hitbox_off.y) + absf(geo->hitbox_off.z) : 0.0f;
    const V3 travel = body.vel * c::DT;
    const float pad = reach + 0.08f;  // ray backoff plus margin for the bound's rounding
    if (absf(body.pos.x) + absf(travel.x) + pad < SW_MESH_AABB[0][0] &&
        absf(body.pos.y) + absf(travel.y) + pad < -SW_MESH_AABB[0][4] &&
        body.pos.z - absf(travel.z) - pad > 0.0f &&
        body.pos.z + absf(travel.z) + pad < c::ARENA_Z_UU * c::UU_TO_BT) return false;
    V3 cur = body.pos, to = body.pos + travel;
    if (geo) {
        cur = cur + body.r * geo->hitbox_off;
        Body pred = body;
        integrate_game(pred, body.vel, body.ang, c::DT);
        to = pred.pos + pred.r * geo->hitbox_off;
    }
    const V3 m = to - cur;
    const float m2 = len2(m);
    if (!(m2 > c::SWEEP_MIN_MOVE_BT * c::SWEEP_MIN_MOVE_BT)) return false;
    // Normalize the ray direction before applying the backoff, as Bullet's predictive cast does.
    const V3 from = cur - (m * (1.0f / sqrtf(m2))) * 0.04f;
    const V3 delta = to - from;
    RayHit hit{1.0f, v3s(0.0f), false};
    const int cell = cell_of(ad, cur);
    for (uint32_t i = ad.cell_off[cell]; i < ad.cell_off[cell + 1]; i++) {
        Tri tri = ad.tris[ad.cell_idx[i]];
        V3 org = v3s(0.0f);
        if (tri.flags & TF_ORIGIN_GOAL) {
            V3 v[3]; goal_tri_local(tri, ad.goal_ly, v); org = tri_origin(tri);
            tri.p0 = v[0]; tri.p1 = v[1]; tri.p2 = v[2];
        }
        ray_tri(from - org, to - org, tri, &hit);
    }
    for (int i = 0; i < NUM_PLANES; i++) {
        V3 o, n; ray_plane_at(i, &o, &n); ray_plane_game(from, to, o, n, &hit);
    }
    if (!hit.ok || !(hit.frac < 1.0f)) return false;
    const V3 d = delta * hit.frac;
    const float dl = sqrtf(len2(d));
    const float shorten = dl > 0.04f ? dl - 0.04f : 0.0f;
    const V3 ds = (d * (1.0f / dl)) * shorten;  // normalize first, then shorten
    cp.n = hit.n;
    cp.d = ((delta.y - ds.y) * hit.n.y + (delta.x - ds.x) * hit.n.x) + (delta.z - ds.z) * hit.n.z;
    return true;
}

// What the solver's rows are built from: every manifold's points. A world manifold's are a run of its body's array, in
// manifold order, as many as the manifold holds.
template <int NC>
struct RowSource {
    static constexpr int NP = NC * (NC - 1) / 2;
    const Body* bodies;
    const PredictiveHit* predictive;
    const ContactPoint* ball_world;
    int ball_world_n;  // zero for a sleeping ball: its world manifolds give no points
    const ContactPoint (*car_world)[CAR_WORLD_SLOTS];
    const ContactPoint (*car_ball)[CAR_BALL_SLOTS];
    const int* car_ball_n;
    const ContactPoint (*car_car)[CAR_CAR_SLOTS];  // slot q of pair p is written only for q < pair_man[p].n
    bool any_car_car;                              // some pair holds a point
};

// The rows in the sweep's manifold order: world, ball-car and car-car manifolds as the dispatcher's sorted array has them,
// each manifold's points in cache order. A manifold with points the sweep does not list (its model of the pair cache and
// RLMAX's contact data disagree) follows, in the order the rows had before the sweep, so no contact is lost. Out of line,
// so the tick's registers stay as they were.
template <int NC>
RLMAX_COLD void emit_rows(TickSolver<NC>& sv, const RowSource<NC>& r, const ArenaState<NC>& s, const SweepKey<NC>* order, int n,
                          bool& ball_in_solver) {
    constexpr int BIT_BW = NC * CAR_WORLD_MANIFOLDS, BIT_BC = BIT_BW + BALL_WORLD_MANIFOLDS, BIT_CC = BIT_BC + NC;
    const Body* bodies = r.bodies;
    uint64_t done = 0;
    uint32_t done_pairs = 0;  // eight-car pair bits do not fit in done
    auto emit_cw = [&](int i, int q) {
        done |= 1ull << (i * CAR_WORLD_MANIFOLDS + q);
        if (q >= (int)s.cars[i].wman.n) return;
        int at = 0;
        for (int p = 0; p < q; p++) at += (int)s.cars[i].wman.m[p].n;
        for (int j = 0; j < (int)s.cars[i].wman.m[q].n; j++) sv.add_point(r.car_world[i][at + j], bodies[1 + i].pos, v3s(0));
    };
    auto emit_bw = [&](int q) {
        done |= 1ull << (BIT_BW + q);
        if (r.ball_world_n == 0 || q >= (int)s.ball_wman.n) return;
        int at = 0;
        for (int p = 0; p < q; p++) at += (int)s.ball_wman.m[p].n;
        for (int j = 0; j < (int)s.ball_wman.m[q].n; j++) sv.add_point(r.ball_world[at + j], bodies[0].pos, v3s(0));
    };
    auto emit_cb = [&](int i) {
        done |= 1ull << (BIT_BC + i);
        for (int q = 0; q < r.car_ball_n[i]; q++) {
            ball_in_solver = true;  // a sleeping ball joins the solver when a car touches it
            sv.add_point(r.car_ball[i][q], bodies[1 + i].pos, bodies[0].pos);
        }
    };
    auto emit_cc = [&](int i, int j, int p) {
        if constexpr (NC == 8) done_pairs |= 1u << p;
        else done |= 1ull << (BIT_CC + p);
        for (int q = 0; q < (int)s.pair_man[p].n; q++) sv.add_point(r.car_car[p][q], bodies[r.car_car[p][q].a].pos, bodies[r.car_car[p][q].b].pos);
    };
    for (int e = 0; e < n; e++) {
        const int lo = sw_lo_for<NC>(order[e]), hi = sw_hi_for<NC>(order[e]);
        if (lo == 0) {
            const int b = hi - SW_BALL;
            const PredictiveHit& hit = r.predictive[b];
            const ContactPoint cp{(int8_t)b, -1, false, true, bodies[b].pos, v3s(0.0f), hit.n, hit.d,
                                  b ? c::CAR_FRICTION : c::BALL_FRICTION, 0.0f};
            sv.add_point(cp, bodies[b].pos, v3s(0.0f));
        } else if (lo <= SW_STATICS) {
            const uint32_t group = lo <= 16 ? (uint32_t)(lo - 1) : 0x100u + (uint32_t)(lo - 17);
            if (hi == SW_BALL) {
                for (int q = 0; q < (int)s.ball_wman.n; q++)
                    if (s.ball_wman.obj[q] == group) emit_bw(q);
            } else {
                const int i = hi - SW_BALL - 1;
                for (int q = 0; q < (int)s.cars[i].wman.n; q++)
                    if (s.cars[i].wman.obj[q] == group) emit_cw(i, q);
            }
        } else if (lo == SW_BALL) {
            emit_cb(hi - SW_BALL - 1);
        } else if (r.any_car_car) {  // no pair holds a point: emit_cc adds nothing
            const int i = lo - SW_BALL - 1, j = hi - SW_BALL - 1;
            emit_cc(i, j, i * (NC - 1) - i * (i - 1) / 2 + (j - i - 1));
        }
    }
    for (int i = 0; i < NC; i++)
        for (int q = 0; q < CAR_WORLD_MANIFOLDS; q++)
            if (!(done >> (i * CAR_WORLD_MANIFOLDS + q) & 1)) emit_cw(i, q);
    for (int i = 0; i < NC; i++)
        if (!(done >> (BIT_BC + i) & 1)) emit_cb(i);
    if (r.any_car_car) {
        int p = 0;
        for (int i = 0; i < NC; i++)
            for (int j = i + 1; j < NC; j++, p++)
                if constexpr (NC == 8) {
                    if (!(done_pairs >> p & 1u)) emit_cc(i, j, p);
                } else if (!(done >> (BIT_CC + p) & 1)) emit_cc(i, j, p);
    }
    for (int q = 0; q < BALL_WORLD_MANIFOLDS; q++)
        if (!(done >> (BIT_BW + q) & 1)) emit_bw(q);
}

// The solve itself, on a caller-owned solver: its rows are the largest object of the tick.
template <int NC, bool Seeded>
RLMAX_HD void solve_contacts_on(TickSolver<NC>& sv, ArenaState<NC>& s, Body* bodies, const RowSource<NC>& src,
                                const SweepKey<NC>* sw_order, int sw_n, bool wheels_woke_ball,
                                bool ball_was_sleeping) {
    constexpr int NB = NC + 1;
    BallState& bs = s.ball;
    sv.reset();
    bool in_solver[NB];
    for (int i = 0; i < NB; i++) {
        const Body& b = bodies[i];
        SBody& sb = sv.bodies[i];
        sb.lin = b.vel;
        sb.ang = b.ang;
        sb.ext_lin = b.acc_lin;
        sb.ext_ang = b.acc_ang;
        sb.dlin = sb.dang = sb.push = sb.turn = v3s(0.0f);
        sb.inv_i = b.inv_i;
        sb.inv_mass = b.inv_mass;
        sb.handle = s.sweep.handle[i];
        sb.dynamic = b.contact_response;
        in_solver[i] = b.active;
    }
    in_solver[0] = in_solver[0] || wheels_woke_ball;
    emit_rows<NC>(sv, src, s, sw_order, sw_n, in_solver[0]);
    sv.emit_synthetics();
    sv.solve();
    for (int i = 0; i < NB; i++) {
        if (!in_solver[i]) continue;
        Body& b = bodies[i];
        const SBody& sb = sv.bodies[i];
        V3 lin = sb.lin + sb.dlin;
        V3 ang = sb.ang + sb.dang;
        if (len2(sb.push) != 0.0f || len2(sb.turn) != 0.0f) {
            // world transform only; the inertia tensor updates at integration
            integrate_game(b, sb.push, sb.turn * c::SPLIT_IMPULSE_TURN_ERP, c::DT);
            if constexpr (Seeded) {
                if (i == 0 && (s.seed_mask & 1u)) {
                    // Split integration moves the transform but does not update inertia.
                    s.seed_basis[0] = b.r;
                    s.seed_inv[0] = b.inv_i;
                    s.seed_mask |= 1u << 20;
                }
            }
        }
        b.vel = lin + sb.ext_lin;
        b.ang = ang + sb.ext_ang;
        if (i == 0 && ball_was_sleeping && (!is_zero(b.vel) || !is_zero(b.ang))) {
            bs.sleeping = BALL_ACTIVE;  // a touch activates the ball (activate() clears the timer); sleeps again after 2 s
            bs.deactivation_time = 0.0f;
        }
    }

}

// Keep solver rows and bodies in a separate frame, entered only when contact candidates exist.
template <int NC, bool Seeded>
RLMAX_COLD void solve_contacts(ArenaState<NC>& s, Body* bodies, const RowSource<NC>& src,
                               const SweepKey<NC>* sw_order, int sw_n, bool wheels_woke_ball,
                               bool ball_was_sleeping) {
    TickSolver<NC> sv;
    solve_contacts_on<NC, Seeded>(sv, s, bodies, src, sw_order, sw_n, wheels_woke_ball, ball_was_sleeping);
}
// The same solve on a solver held in per-arena global memory (the staged path), so no kernel frame carries the rows.
template <int NC, bool Seeded>
RLMAX_COLD void solve_contacts_ext(TickSolver<NC>& sv, ArenaState<NC>& s, Body* bodies, const RowSource<NC>& src,
                                   const SweepKey<NC>* sw_order, int sw_n, bool wheels_woke_ball,
                                   bool ball_was_sleeping) {
    solve_contacts_on<NC, Seeded>(sv, s, bodies, src, sw_order, sw_n, wheels_woke_ball, ball_was_sleeping);
}

// Point i of a world manifold on A, the body (Bullet m_positionWorldOnA). Out of line: it only feeds the solver's lever arm.
RLMAX_COLD V3 man_point_a(const Body& a, const Manifold& m, int i) { return man_to_world(&a, m.p[i].la); }

// The cars' geometry, derived from the body presets (body_geo: the steering knots alone are eight sincos / atan2 / divisions
// a car), kept across the ticks of one launch: a car's geometry is a function of its preset, and the preset changes only
// at a respawn (respawn_car) or an upload, never between two launches' states. The key is the clamped preset id
// (body_preset reads ids past the table as 0), so no state value equals the invalid key.
template <int NC>
struct GeoCache {
    CarGeo g[NC];
    uint32_t key[NC];
    RLMAX_HD void reset() {
        for (int i = 0; i < NC; i++) key[i] = 0xFFFFFFFFu;
    }
};

// Out of line: runs once per car per launch, not per tick.
RLMAX_COLD void geo_fill(CarGeo& g, uint32_t key) { g = body_geo(key); }

// The game's vehicle order: a demolished car's vehicle leaves the list by swap with the
// last present one, a respawned car's is appended, and absent cars sit after the present. Run at the start of a tick: the
// cars respawned last tick are appended first, then those demolished last tick are removed, each in slot order.
template <int NC>
RLMAX_HD void update_visit_order(ArenaState<NC>& s) {
    uint32_t demoed = 0;
    for (int i = 0; i < NC; i++) demoed |= (uint32_t)fl(s.cars[i], CF_IS_DEMOED) << i;
    uint32_t off = s.visit >> 24;
    if (demoed == off) return;
    int o[NC], present = 0;
    for (int k = 0; k < NC; k++) {
        o[k] = visit_slot<NC>(s.visit, k);
        present += (off >> k & 1u) == 0;
    }
    auto place = [&](int slot) { int p = 0; while (o[p] != slot) p++; return p; };
    for (int i = 0; i < NC; i++) {
        if (!((off & ~demoed) >> i & 1u)) continue;
        int p = place(i);
        for (; p > present; p--) o[p] = o[p - 1];
        o[present++] = i;
        off &= ~(1u << i);
    }
    for (int i = 0; i < NC; i++) {
        if (!((demoed & ~off) >> i & 1u)) continue;
        const int p = place(i), q = --present;
        o[p] = o[q];
        o[q] = i;
        off |= 1u << i;
    }
    uint32_t v = off << 24;
    for (int k = 0; k < NC; k++) v |= (uint32_t)(o[k] ^ k) << ((NC == 8 ? 3 : 4) * k);
    s.visit = v;
}

template <int NC, bool HasPsyclops, bool Seeded>
RLMAX_HD void tick_impl(ArenaState<NC>& s, const ArenaData& ad, EpaScratch& scratch, GeoCache<NC>& gc) {
    RLMAX_PROF_BEGIN
#ifdef __CUDA_ARCH__
    // the sweep reads its state late in the tick, from an arena 3.5 KB from its warp neighbours' (one line each, DRAM
    // latency): ask L2 for its lines now, so the reads there hit (P3 §6, the sweep's sub-profile)
    for (int o = 0; o < (int)sizeof(s.sweep); o += 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<const char*>(&s.sweep) + o));
#endif
    const TickConsts k = tick_consts();
    for (int i = 0; i < NC; i++) {
        const uint32_t b = s.cars[i].body, key = b < (uint32_t)NUM_BODY_PRESETS ? b : 0u;
        if (key != gc.key[i]) {
            geo_fill(gc.g[i], key);
            gc.key[i] = key;
        }
    }
    const CarGeo* geo = gc.g;
    constexpr int NB = 1 + NC;
    constexpr int NP = NC * (NC - 1) / 2;
    Body bodies[NB];

    RLMAX_PROF(0)
    // ---- bodies from state
    BallState& bs = s.ball;
    const bool ball_uploaded = bs.sleeping == BALL_UPLOADED;
    if (ball_uploaded) resolve_uploaded_ball(bs);
    bodies[0] = make_body(bs.pos, bs.rot, bs.vel, bs.ang_vel, c::BALL_MASS_BT, k.ball_inv_inertia);
    if (ball_uploaded && bs.sleeping == BALL_SLEEPING) prime_uploaded_ball_planes(s.ball_wman, bodies[0], k.ball_radius, k.ball_threshold);
    // Bullet retains the pre-sleep tensor through wake; the rounded quaternion
    // alone cannot reconstruct its float bits after a device handoff.
    if (bs.sleeping == BALL_SLEEPING) bodies[0].inv_i = bs.sleep_inv_i;
    RLMAX_SEED_BALL(bodies[0], bs.sleeping == BALL_SLEEPING);
    bodies[0].active = bs.sleeping != BALL_SLEEPING;
    for (int i = 0; i < NC; i++) {
        CarState& cs = s.cars[i];
        bodies[1 + i] = make_body(cs.pos, cs.rot, cs.vel, cs.ang_vel, c::CAR_MASS_BT, geo[i].inv_inertia_local);
        RLMAX_SEED_BASIS(i, bodies[1 + i]);
        bool demo = fl(cs, CF_IS_DEMOED);
        bodies[1 + i].active = !demo;
        bodies[1 + i].contact_response = !demo;
    }
    if constexpr (Seeded) {
        const uint32_t mask = s.seed_mask;
        for (int i = 0; i < NB; ++i) {
            const uint32_t bit = 1u << i;
            if (!(mask & bit)) continue;
            Body& body = bodies[i];
            body.r = s.seed_basis[i];
            body.inv_i = (mask & (bit << 20)) ? s.seed_inv[i] : world_inv_inertia(body.r, body.inv_i_local);
        }
    }

    RLMAX_PROF(1)
    // ---- limit velocities, then quantise
    V3 pre_q_pos[NB];
    for (int i = 0; i < NB; i++) pre_q_pos[i] = bodies[i].pos;
    for (int i = 0; i < NC; i++) {
        // the speed limit in uu, and its result is what the codec rounds
        Body& cb = bodies[1 + i];
        V3 vu = cb.vel * c::BT_TO_UU;
        if (len2(vu) > c::CAR_MAX_SPEED_UU * c::CAR_MAX_SPEED_UU) {
            vu = normalize_or_zero(vu) * c::CAR_MAX_SPEED_UU;
            cb.vel = vu * c::UU_TO_BT;
        }
        if (len2(cb.ang) > c::CAR_MAX_ANG_SPEED * c::CAR_MAX_ANG_SPEED) cb.ang = normalize_or_zero(cb.ang) * c::CAR_MAX_ANG_SPEED;
        if (quirk::QUANTIZE_PHYS) quantize_body(cb, vu);
    }
    limit_vels(bodies[0], c::BALL_MAX_SPEED_UU * c::UU_TO_BT, c::BALL_MAX_ANG_SPEED);
    if (quirk::QUANTIZE_PHYS) quantize_body(bodies[0], bodies[0].vel * c::BT_TO_UU);

    // Callback rotations are saved only for cars with contacts, before solving.
    CarSnap snap[NC];
    const V3 ball_snap_vel_uu = bodies[0].vel_uu;

    RLMAX_PROF(2)
    // ---- flip Z damping pre-pass
    for (int i = 0; i < NC; i++) {
        CarState& cs = s.cars[i];
        if (fl(cs, CF_IS_DEMOED) || fl(cs, CF_ON_GROUND) || !fl(cs, CF_IS_FLIPPING)) continue;
        if (!(cs.flip_time >= c::FLIP_Z_DAMP_START && cs.flip_time <= c::FLIP_TORQUE_TIME)) continue;
        Body& rb = bodies[1 + i];
        if (rb.vel.z < 0.0f || cs.flip_time < c::FLIP_Z_DAMP_END) {
            // an impulse through the mass, from the uu velocity the codec wrote
            const float dv_uu = (-rb.vel_uu.z) * c::FLIP_Z_DAMP_120;
            if (absf(dv_uu) >= 1.0e-4f) {
                const float impulse = (dv_uu * c::UU_TO_BT) * (1.0f / rb.inv_mass);
                RLMAX_IMP_VZ(i, IL_DTORQUE, rb, rb.vel.z = rb.vel.z + impulse * rb.inv_mass);
            }
        }
    }

    RLMAX_PROF(3)
    // ---- car pre-tick
    CarTickOut outs[NC];
    bool pre_ticked[NC];
    CarCtx<NC> ctx{&ad, geo, bodies, k.ball_radius, (s.sweep.hdr & 2u) ? s.sweep.box : nullptr};
    update_visit_order<NC>(s);
    for (int vk = 0; vk < NC; vk++) {
        const int i = visit_slot<NC>(s.visit, vk);
        CarState& cs = s.cars[i];
        pre_ticked[i] = false;
        if (fl(cs, CF_IS_DEMOED)) {
            cs.demo_respawn_timer = fmax_(cs.demo_respawn_timer - c::DT, 0.0f);
            if (cs.demo_respawn_timer != 0.0f) {
                bodies[1 + i].active = false;  // hidden until it respawns (DisableSimulation), no contact response
                bodies[1 + i].contact_response = false;
                continue;
            }
            // the respawned car takes this tick as any other: quantised from its spawn pose, then its pre-tick
            if constexpr (Seeded) s.seed_mask &= ~((1u << (1 + i)) | (1u << (21 + i)));
            respawn_car<NC>(cs, s.rng);
#ifdef RLMAX_FULL_STATE
            fs_respawn_override<NC>(cs, i);
#endif
            Body& spawned = bodies[1 + i];
            spawned = make_body(cs.pos, cs.rot, cs.vel, cs.ang_vel, c::CAR_MASS_BT, geo[i].inv_inertia_local);
            spawned.inv_i = world_inv_inertia(respawn_inertia_basis(cs.team), spawned.inv_i_local);
            const V3 spawn_pos = spawned.pos;
#if !defined(RLMAX_FULL_STATE)
            pre_q_pos[1 + i] = spawn_pos;
#endif
            if (quirk::QUANTIZE_PHYS) quantize_body(spawned, spawned.vel_uu);
        }
        if constexpr (HasPsyclops) {
            if (cs.body == BODY_PSYCLOPS) car_pre_tick<NC, true>(i, cs, ctx, outs[i]);
            else car_pre_tick<NC>(i, cs, ctx, outs[i]);
        } else car_pre_tick<NC>(i, cs, ctx, outs[i]);
        pre_ticked[i] = true;
    }

    RLMAX_PROF(4)
    // ---- gravity and damping
    for (int i = 0; i < NB; i++) {
        bool active = i == 0 ? bs.sleeping != BALL_SLEEPING : bodies[i].active;
        if (!active) continue;
        if (i == 0) {
            bodies[i].acc_lin += k.gravity * c::DT;
        } else {
            // Bullet's integrateVelocities: (total force x inverse mass) x dt, gravity as a force
            Body& b = bodies[i];
            const V3 total = b.force + v3(0.0f, 0.0f, (c::GRAVITY_Z_UU * c::UU_TO_BT) * b.mass);
            b.acc_lin += (total * b.inv_mass) * c::DT;
            if (!is_zero(b.torque)) b.acc_ang += (b.inv_i * b.torque) * c::DT;
        }
    }
    if (bs.sleeping != BALL_SLEEPING) bodies[0].vel = bodies[0].vel * BALL_DAMPING_PER_TICK;

    RLMAX_PROF(5)
    // a wheel on the ball activates it, as the ball's applyImpulse does: a sleeping one is solved this tick with its
    // world contacts, without this tick's gravity, and either way its deactivation timer restarts
    bool wheels_woke_ball = false;
    V3 predict_lin[NB], predict_ang[NB];  // the velocities the broadphase predicts with: after the pre-ticks, before the wheels
    // ---- vehicles, inside the step (bin 2b8fc0 step d, updateActions): after gravity (a force, not yet in the
    // velocities), the speed clamp and damping (step b), before collision detection; every car's components have run
    if (quirk::VEHICLE_IN_STEP) {
        for (int i = 0; i < NC; i++)
            if (bodies[1 + i].active) limit_vels(bodies[1 + i], c::CAR_MAX_SPEED_UU * c::UU_TO_BT, c::CAR_MAX_ANG_SPEED);
        if (bs.sleeping != BALL_SLEEPING)
            limit_vels(bodies[0], c::BALL_MAX_SPEED_UU * c::UU_TO_BT, c::BALL_MAX_ANG_SPEED);
        for (int i = 0; i < NB; i++) { predict_lin[i] = bodies[i].vel; predict_ang[i] = bodies[i].ang; }
        uint32_t visited = 0;
        for (int vk = 0; vk < NC; vk++) {
            const int i = visit_slot<NC>(s.visit, vk);
            CarState& cs = s.cars[i];
            visited |= 1u << i;
            if (!pre_ticked[i]) continue;  // demoed, or respawned this tick (frozen)
            if constexpr (HasPsyclops) {
                if (cs.body == BODY_PSYCLOPS) vehicle_update<NC, true>(i, cs, ctx, cs.handbrake_val, outs[i].real_throttle, outs[i]);
                else vehicle_update<NC>(i, cs, ctx, cs.handbrake_val, outs[i].real_throttle, outs[i]);
            } else vehicle_update<NC>(i, cs, ctx, cs.handbrake_val, outs[i].real_throttle, outs[i]);
            wheels_woke_ball = wheels_woke_ball || (outs[i].pushed & PUSHED_BALL) != 0;
            // the wheels' ground-stick pushed a car that has yet to take its turn: its drive follows the new velocity
            if (outs[i].pushed != 0)
                for (int j = 0; j < NC; j++)
                    if ((outs[i].pushed >> j & 1u) && !(visited >> j & 1u) && pre_ticked[j]) car_redrive<NC, HasPsyclops>(j, s.cars[j], ctx, outs[j]);
        }
    }

    if (wheels_woke_ball) {
        bs.deactivation_time = 0.0f;
        if (bs.sleeping == BALL_WANTS_DEACTIVATION) bs.sleeping = BALL_ACTIVE;
    }

    RLMAX_PROF(6)
    // ---- collision detection at the current pose
    // the solver's points, occupied slots only, in slot order (the order the rows are built in): writing every slot
    // every tick was the largest share of the tick's local stores (P3 §3.7); add_point skips an empty slot, so the
    // rows are the same
    ContactPoint ball_world[BALL_WORLD_SLOTS];
    ContactPoint car_world[NC][CAR_WORLD_SLOTS];
    ContactPoint car_ball[NC][CAR_BALL_SLOTS];
    int ball_world_n = 0, car_world_n[NC], car_ball_n[NC];
    bool car_ball_touch[NC];
    ContactPoint car_car[NP > 0 ? NP : 1][CAR_CAR_SLOTS];
    bool any_car_car = false;  // some pair holds a point: the callbacks act only on cp.ok, so with none they are skipped
    bool car_world_hit[NC];
    V3 car_world_normal[NC];
    RLMAX_PROF(7)
    for (int i = 0; i < NC; i++) {
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
            // the convex-concave callback takes only triangles that touch the box's bounding box (TestTriangleAgainstAabb2)
            V3 bmn, bmx;
            obb_aabb(ob, &bmn, &bmx);
            const V3 ext = obb_extent(ob);
            int cell = cell_of(ad, ob.c);
            for (uint32_t t = ad.cell_off[cell]; t < ad.cell_off[cell + 1]; t++) {
                const uint32_t tri_index = ad.cell_idx[t];
                const Tri& tri = ad.tris[tri_index];
                // a goal triangle is tested in its body's frame, as the game's tree query is: the box's y less the origin against the game's local vertex y
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
        // sum contact normals in static-object order, then normalize once.
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
    RLMAX_PROF(8)
    // car vs ball: the pair's persistent manifold (A = ball, B = car; manifold.h); the extra-hit callback below fires
    // whenever the refreshed manifold holds a point, fresh or retained
    for (int i = 0; i < NC; i++) {
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
            const float max_dist = cob.margin + k.ball_radius + thr;
            const GjkSphereResult gs = gjk_box_sphere(cob.h, cob.margin, cob.r, cob.c, ba.pos, k.ball_radius, max_dist * max_dist);
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
            // the solver's first body is the car (the car's term leads each row's
            // delta), so its normal points at the car; the manifold keeps the ball as A
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
    {
        // ball vs static (special points): one manifold per (ball, mesh component or plane) pair (manifold.h). The
        // sphere's pairs are plain shapes: this tick's points added, then refreshed. A sleeping ball and the static
        // world are both inactive: no pair runs and the manifolds stay as they are, unless a car touches the ball:
        // it wakes after applyGravity, joins the car's island and is solved this tick with its world contacts too
        bool car_touch = false;
        for (int i = 0; i < NC; i++) car_touch = car_touch || car_ball_n[i] > 0;
        // a car's touch reaches the world only if the ball's manifolds hold something; a wheel's always does
        const bool awake = bs.sleeping != BALL_SLEEPING || wheels_woke_ball || (car_touch && s.ball_wman.n != 0);
        auto& w = s.ball_wman;
        if (awake) {
            const Body& rb = bodies[0];
            const float thr = k.ball_threshold;
            // a goal triangle's point on the world is kept as an offset from the goal body's origin
            auto add = [&](const Contact& f, V3 org, V3 on_ball) {
                if (!f.ok || f.d > thr) return;
                Manifold* m = wman_get(w, f.group);
                if (m) man_add_point(*m, MPoint{man_to_local(&rb, on_ball), f.pb - org, f.n, f.d}, thr);
            };
            V3 bc = rb.pos;
            for (int p = 0; p < NUM_PLANES + 2; p++) {  // the goal floors too: the game's planes 104 and 105
                Contact cp = sphere_plane(bc, rb.r, k.ball_radius, plane_xf(p), thr);
                cp.group = 0x100u + (uint32_t)p;
                add(cp, plane_xf(p).o, cp.pb + cp.n * cp.d);
            }
            // the convex-concave callback takes only triangles that touch the sphere's bounding box (TestTriangleAgainstAabb2)
            const V3 br = v3s(k.ball_radius + c::SPHERE_BOUND_PAD_BT);
            int cell = cell_of(ad, bc);
            for (uint32_t t = ad.cell_off[cell]; t < ad.cell_off[cell + 1]; t++) {
                const Tri& tri = ad.tris[ad.cell_idx[t]];
                if (quirk::TRI_AABB_GATE_BALL) {
                    // a goal triangle is tested in its body's frame, the game's local vertex y against the centre less the origin
                    if (tri.flags & TF_ORIGIN_GOAL) {
                        V3 lv[3];
                        goal_tri_local(tri, ad.goal_ly, lv);
                        const float cy = bc.y - tri_origin(tri).y;
                        if (!tri_aabb_overlap(lv[0], lv[1], lv[2], v3(bc.x - br.x, cy - br.y, bc.z - br.z), v3(bc.x + br.x, cy + br.y, bc.z + br.z))) continue;
                    } else if (!tri_aabb_overlap(tri.p0, tri.p1, tri.p2, bc - br, bc + br)) {
                        continue;
                    }
                }
                Contact cc = sphere_tri(tri, bc, k.ball_radius, thr, ad.goal_ly);
                cc.group = tri_mesh(tri);
                // the edge adjustment runs as the point is added (contact-added callback), before the manifold sees it
                // the point on the ball is kept as it was found: the adjustment turns the normal and moves only the point on the world
                const V3 on_ball = cc.pb + cc.n * cc.d;
                if (cc.ok) adjust_internal_edge(tri, &cc, ad.goal_ly);
                add(cc, tri_origin(tri), on_ball);
            }
            V3 wpa[MANIFOLD_CAPACITY], wpb[MANIFOLD_CAPACITY];
            for (int q = 0; q < BALL_WORLD_MANIFOLDS; q++)
                if (q < (int)w.n) {
                    const V3 org = mesh_origin(w.obj[q]);
                    // a plane's point comes back through the plane's transform, whose rounding is what leaves a rolling ball a distance
                    const bool pl = w.obj[q] >= 0x100u;
                    const PlaneXf px = plane_xf(pl ? (int)(w.obj[q] - 0x100u) : 0);
                    man_refresh(w.m[q], &rb, nullptr, thr, wpa, wpb, pl ? &px : nullptr, w.obj[q] - 4u < 4u ? &org : nullptr);
                }
            wman_release_empty(w);
        }
        // a sleeping untouched ball's manifolds are kept but give the solver no points this tick
#pragma unroll
        for (int q = 0; q < BALL_WORLD_MANIFOLDS; q++) {
            if (!awake || q >= (int)w.n) break;
#pragma unroll
            for (int pi = 0; pi < MANIFOLD_CAPACITY; pi++) {
                if (pi >= (int)w.m[q].n) break;
                const V3 org = mesh_origin(w.obj[q]);
                const bool pl = w.obj[q] >= 0x100u;
                const PlaneXf px = plane_xf(pl ? (int)(w.obj[q] - 0x100u) : 0);
                const Contact bw = man_point(w.m[q], nullptr, pi, pl ? &px : nullptr, w.obj[q] - 4u < 4u ? &org : nullptr);
                ContactPoint& cp = ball_world[ball_world_n++];
                cp.ok = true;
                cp.a = 0;
                cp.b = -1;
                cp.pb = bw.pb;
                cp.n = bw.n;
                cp.d = bw.d;
                cp.pa = man_point_a(bodies[0], w.m[q], pi);  // the manifold's point on A (Bullet m_positionWorldOnA)
                cp.friction = fmin_(c::BALL_FRICTION, c::ARENA_FRICTION);
                cp.restitution = fmax_(c::BALL_RESTITUTION, c::ARENA_RESTITUTION);
                cp.special = true;
            }
        }
    }
    // the broadphase sweep (sweep.h) from this tick's boxes: the solver's manifold order, and which car pairs hold a
    // manifold (the dispatcher's, made when the pair's compound child algorithm is, destroyed with it)
    PredictiveHit predictive[NB];
    uint32_t predictive_mask = 0;
    for (int b = 0; b < NB; b++) {
        if (bodies[b].active && predictive_world(ad, bodies[b], b ? &geo[b - 1] : nullptr, predictive[b])) {
            predictive_mask |= 1u << b;
            if (b && car_world_hit[b - 1]) {
                const auto& w = s.cars[b - 1].wman;
                V3 sum = v3s(0.0f);
                bool added = false;
                uint32_t last = 0;
                for (uint32_t n = 0; n < w.n; n++) {
                    int best = -1;
                    for (int q = 0; q < (int)w.n; q++) {
                        if (n && w.obj[q] <= last) continue;
                        if (best < 0 || w.obj[q] < w.obj[best]) best = q;
                    }
                    if (best < 0) break;
                    last = w.obj[best];
                    if (!added && last >= 0x100u) { sum += predictive[b].n; added = true; }
                    for (int pi = 0; pi < (int)w.m[best].n; pi++) sum += w.m[best].p[pi].n;
                }
                if (!added) sum += predictive[b].n;
                const float l2 = len2(sum);
                if (l2 >= 1e-8f) car_world_normal[b - 1] = sum * (1.0f / sqrtf(l2));
            }
        }
    }
    bool known_solver_points = predictive_mask != 0 || ball_world_n != 0;
    for (int i = 0; i < NC && !known_solver_points; i++)
        known_solver_points = car_world_n[i] != 0 || car_ball_n[i] != 0;
    const uint32_t sw_had = sweep_car_pairs<NC>(s.sweep);
    SweepKey<NC> sw_order[sweep_man_cap<NC>()];
    const int sw_n = sweep_tick<NC>(s.sweep, bodies, predict_lin, predict_ang, geo, k.ball_radius, s.visit, predictive_mask, sw_order, known_solver_points, ball_held_planes(s.ball_wman));
    const uint32_t sw_has = sweep_car_pairs<NC>(s.sweep);
    RLMAX_PROF(18)
    RLMAX_PROF(9)
    {
        int p = 0;
        for (int i = 0; i < NC; i++) {
            for (int j = i + 1; j < NC; j++, p++) {
                // the pair's persistent manifold: this tick's box-box points added, then refreshed (manifold.h)
                // A is the pair's higher broadphase handle
                const int ia = s.sweep.handle[1 + i] > s.sweep.handle[1 + j] ? i : j, ib = i + j - ia;
                const Body& ba = bodies[1 + ia];
                const Body& bb = bodies[1 + ib];
                Manifold& m = s.pair_man[p];
                V3 wpa[MANIFOLD_CAPACITY], wpb[MANIFOLD_CAPACITY];
                // a pair holds its points only as long as its manifold lives in the sweep: none made (or destroyed) this tick
                // means no collision, one made this tick starts empty
                if (!(ba.contact_response && bb.contact_response && (sw_has >> p & 1u))) {
                    if (m.n) m.n = 0;
                } else {
                    if (!(sw_had >> p & 1u)) m.n = 0;
                    const float thr = fmin_(geo[i].contact_threshold, geo[j].contact_threshold);
                    Contact fresh[CAR_CAR_SLOTS];
                    V3 fresh_a[CAR_CAR_SLOTS];
                    obb_obb<CAR_CAR_SLOTS>(car_obb(ba, geo[ia]), car_obb(bb, geo[ib]), thr, fresh, fresh_a);
                    // matching selects by the lower-handle car's local point.
                    man_collide(m, &ba, &bb, fresh, thr, quirk::COMPOUND_REFRESH_FIRST, wpa, wpb, true, fresh_a);
                }
                static_assert(CAR_CAR_SLOTS == MANIFOLD_CAPACITY, "one slot per manifold point");
                const int held = (int)m.n;
                if (held) {
                    snap[ia].q = ba.rot;
                    snap[ib].q = bb.rot;
                    any_car_car = true;
                }
                // only the held points are written (the readers stop at pair_man[p].n): writing the four slots of every pair
                // every tick was 15 pairs x 64 local stores at 3v3, for a pair that holds a point 0.5% of the ticks
                for (int q = 0; q < held; q++) {
                    ContactPoint& cp = car_car[p][q];
                    cp.ok = true;
                    cp.a = 1 + ia;
                    cp.b = 1 + ib;
                    cp.pb = wpb[q];
                    cp.n = m.p[q].n;
                    cp.d = m.p[q].d;
                    cp.pa = wpa[q];  // the manifold's point on A (Bullet m_positionWorldOnA), not B plus the normal depth
                    cp.friction = c::CAR_CAR_FRICTION;
                    cp.restitution = c::CAR_CAR_RESTITUTION;
                    cp.special = false;
                }
            }
        }
    }

    RLMAX_PROF(10)
    // ---- ball sleeping (island rule: no active car's AABB overlaps the ball's)
    if (bs.sleeping == BALL_WANTS_DEACTIVATION) {
        bool island_active = false;
        V3 bmin = bodies[0].pos - v3s(k.ball_radius + 0.08f), bmax = bodies[0].pos + v3s(k.ball_radius + 0.08f);
        for (int i = 0; i < NC; i++) {
            if (!bodies[1 + i].contact_response) continue;
            Obb ob = car_obb(bodies[1 + i], geo[i]);
            V3 fh = obb_full_half(ob);
            V3 ext = v3(dot(vabs(v3(ob.r.c0.x, ob.r.c1.x, ob.r.c2.x)), fh), dot(vabs(v3(ob.r.c0.y, ob.r.c1.y, ob.r.c2.y)), fh),
                        dot(vabs(v3(ob.r.c0.z, ob.r.c1.z, ob.r.c2.z)), fh));
            V3 cmin = ob.c - ext, cmax = ob.c + ext;
            bool overlap = !(cmin.x > bmax.x || cmax.x < bmin.x || cmin.y > bmax.y || cmax.y < bmin.y ||
                             cmin.z > bmax.z || cmax.z < bmin.z);
            island_active = island_active || overlap;
        }
        if (!island_active) {
            bs.sleeping = BALL_SLEEPING;
            bs.sleep_inv_i = bodies[0].inv_i;
            bodies[0].vel = bodies[0].ang = v3s(0.0f);
            bodies[0].active = false;
        }
    }

    RLMAX_PROF(11)
    // ---- solve
    const bool ball_was_sleeping = bs.sleeping == BALL_SLEEPING;
    const bool has_solver_points = known_solver_points || any_car_car;
    if (has_solver_points) {
        const RowSource<NC> src{bodies, predictive, ball_world, ball_world_n, car_world, car_ball, car_ball_n, car_car, any_car_car};
        solve_contacts<NC, Seeded>(s, bodies, src, sw_order, sw_n, wheels_woke_ball, ball_was_sleeping);
    } else {
        for (int i = 0; i < NB; i++) {
            Body& b = bodies[i];
            if (!b.active && !(i == 0 && wheels_woke_ball)) continue;
            // Match the empty solver's additions, including their signed-zero behavior.
            b.vel = (b.vel + v3s(0.0f)) + b.acc_lin;
            b.ang = (b.ang + v3s(0.0f)) + b.acc_ang;
            if (i == 0 && ball_was_sleeping && (!is_zero(b.vel) || !is_zero(b.ang))) {
                bs.sleeping = BALL_ACTIVE;
                bs.deactivation_time = 0.0f;
            }
        }
    }

    RLMAX_PROF(12)
    // ---- integrate active bodies
    for (int i = 0; i < NB; i++) {
        bool active = i == 0 ? bs.sleeping != BALL_SLEEPING : bodies[i].active;
        if (!active) continue;
        Body& b = bodies[i];
        integrate_game(b, b.vel, b.ang, c::DT);
        if constexpr (Seeded) s.seed_mask &= ~((1u << i) | (1u << (20 + i)));
    }

    RLMAX_PROF(13)
    // ---- ball activation (CanSleep)
    if (bs.sleeping != BALL_SLEEPING) {
        const Body& b = bodies[0];
        bool within = len2(b.vel) < c::BALL_LINEAR_SLEEP_THRESH * c::BALL_LINEAR_SLEEP_THRESH &&
                      len2(b.ang) < c::BALL_ANGULAR_SLEEP_THRESH * c::BALL_ANGULAR_SLEEP_THRESH;
        if (within && !ball_was_sleeping) {  // a tick that woke it restarts the timer at zero
            bs.deactivation_time += c::DT;
            if (bs.deactivation_time > 2.0f && bs.sleeping == BALL_ACTIVE) bs.sleeping = BALL_WANTS_DEACTIVATION;
        } else {
            bs.deactivation_time = 0.0f;
            bs.sleeping = BALL_ACTIVE;
        }
    }

    RLMAX_PROF(14)
    // ---- contact callbacks, after the step (tick-start car and ball state, as the game's callbacks read it)
    V3 bumped_vel[NC];
    // applies pending bumps to velocity in hitter order.
    for (int i = 0; i < NC; i++) bumped_vel[i] = bodies[1 + i].vel;
    for (int i = 0; i < NC; i++) {
        if (!car_ball_touch[i]) continue;
        CarState& cs = s.cars[i];
        // Car_TA.ApplyBallImpactForces takes ForwardDir = GetRotatorAxis(QuatToRotator(OldRBState.Quaternion), 0): yaw
        // and pitch in integer UE3 units (appRound: int(rint(2v + 0.5)) >> 1), the axis rebuilt from UE3's 16384-entry
        // sine table (index angle >> 2)
        // the rotator from the tick-start quaternion by the game's atan2f, the uu state, Ue3SafeNormal at each step
        const Q rot0 = snap[i].q;
        const float y2 = rot0.y + rot0.y, z2 = rot0.z + rot0.z;
        const V3 f = v3(1.0f - (rot0.y * y2 + rot0.z * z2), rot0.x * y2 + rot0.w * z2, rot0.x * z2 - rot0.w * y2);
        auto ue3_round = [](float v) { return ((int)rintf(v + v + 0.5f)) >> 1; };
        const int fwd_yaw = ue3_round(ue3_units(game_atan2f(f.y, f.x)));
        const float fxy = sqrtf(f.y * f.y + f.x * f.x);
        const int fwd_pitch = ue3_round(ue3_units(game_atan2f(f.z, fxy)));
        const float cos_pitch = ue3_sin(fwd_pitch + 16384);
        const V3 car_fwd = v3(cos_pitch * ue3_sin(fwd_yaw + 16384), cos_pitch * ue3_sin(fwd_yaw), ue3_sin(fwd_pitch));
        auto safe_normal = [](V3 v) { const float l2 = len2(v); if (l2 == 1.0f) return v; if (l2 < 1e-8f) return v3s(0.0f); const float sc = 1.0f / sqrtf(l2); return v3(v.x * sc, v.y * sc, v.z * sc); };
        // Both positions still hold their tick-start values until post-tick. Recreate
        // the codec's uu values only when this car actually touched the ball.
        static_assert(quirk::QUANTIZE_PHYS, "ball-hit positions use tick-start quantization");
        auto quantized_pos_uu = [](V3 p) {
            p = p * c::BT_TO_UU;
            return v3(q_pos(p.x, c::QUANT_POS_SCALE), q_pos(p.y, c::QUANT_POS_SCALE), q_pos(p.z, c::QUANT_POS_SCALE));
        };
        const V3 rel_pos = quantized_pos_uu(bs.pos) - quantized_pos_uu(cs.pos);
        const V3 rel_vel = ball_snap_vel_uu - contact_car_velocity_uu(cs);
        float rel_speed = fmin_(len(rel_vel), c::HIT_MAX_DELTA_VEL_UU);
        bool can_accel = !fl(cs, CF_HAS_EXTRA_HIT) || cs.last_extra_hit_tick + 1u < s.tick;
        if (rel_speed > 0.0f && can_accel) {
            V3 hit_dir = safe_normal(rel_pos);
            hit_dir.z *= c::HIT_Z_SCALE;
            hit_dir = safe_normal(hit_dir);
            const V3 adj = (car_fwd * dot(hit_dir, car_fwd)) * (1.0f - c::HIT_FORWARD_SCALE);
            hit_dir = safe_normal(hit_dir - adj);
            const V3 imp = (hit_dir * rel_speed) * curves::BALL_CAR_EXTRA_IMPULSE_FACTOR.eval(rel_speed);
            bodies[0].vel += ((imp * c::UU_TO_BT) * bodies[0].mass) * bodies[0].inv_mass;
            cs.last_extra_hit_tick = s.tick;
            set_fl(cs, CF_HAS_EXTRA_HIT, true);
        }
    }
    if (any_car_car) car_car_callbacks<NC>(s, snap, geo, car_car, bumped_vel, scratch);
    for (int i = 0; i < NC; i++) {
        if (!car_world_hit[i]) continue;
        s.cars[i].world_contact_normal = car_world_normal[i];
        set_fl(s.cars[i], CF_WORLD_CONTACT, true);
    }

    RLMAX_PROF(15)
    // ---- car post-tick and finish
    for (int i = 0; i < NC; i++) {
        CarState& cs = s.cars[i];
        Body& rb = bodies[1 + i];
        if (fl(cs, CF_IS_DEMOED)) {
            // the game keeps the body where it was; the car is hidden until it respawns
            cs.pos = rb.pos;
            cs.rot = rb.rot;
            continue;
        }
        car_post_tick(cs, rb, outs[i]);
        rb.vel = bumped_vel[i];
        cs.pos = rb.pos;
        cs.rot = rb.rot;
        cs.vel = rb.vel;
        cs.ang_vel = rb.ang;
    }
    RLMAX_PROF(16)
    // ---- boost pads, on the cars' end-of-tick state
    // the suite's scorer gives pad boost from the game's recorded pickups, so its test build runs no pads of its own
#if !defined(RLMAX_FULL_STATE)
    pads_update<NC>(s, bodies, geo, pre_q_pos);
#endif

    bs.pos = bodies[0].pos;
    bs.rot = bodies[0].rot;
    bs.vel = bodies[0].vel;
    bs.ang_vel = bodies[0].ang;
    s.tick += 1;

    const float goal_y = (c::GOAL_THRESHOLD_Y_UU + c::BALL_RADIUS_UU) * c::UU_TO_BT;
    if (bs.pos.y > goal_y) s.events |= 1u;   // in orange's goal: blue scored
    if (bs.pos.y < -goal_y) s.events |= 2u;  // in blue's goal: orange scored
    RLMAX_PROF(17)
}

// A seed is uncommon in training. Keep its extra bookkeeping in a cold copy of the tick,
// so an ordinary tick pays only the dispatch test and retains the unseeded solver path.
template <int NC, bool HasPsyclops>
RLMAX_COLD void tick_seeded(ArenaState<NC>& s, const ArenaData& ad, EpaScratch& scratch, GeoCache<NC>& gc) {
    tick_impl<NC, HasPsyclops, true>(s, ad, scratch, gc);
}

template <int NC, bool HasPsyclops = true>
RLMAX_HD void tick(ArenaState<NC>& s, const ArenaData& ad, EpaScratch& scratch, GeoCache<NC>& gc) {
    if (s.seed_mask) tick_seeded<NC, HasPsyclops>(s, ad, scratch, gc);
    else tick_impl<NC, HasPsyclops, false>(s, ad, scratch, gc);
}

// One tick with a fresh geometry cache (the geometry derived again every call).
template <int NC>
RLMAX_HD void tick(ArenaState<NC>& s, const ArenaData& ad, EpaScratch& scratch) {
    GeoCache<NC> gc;
    gc.reset();
    tick<NC>(s, ad, scratch, gc);
}

}  // namespace rlmax
