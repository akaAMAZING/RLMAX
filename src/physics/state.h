// RLMAX arena state: everything needed for the next tick, packed into 32-bit words
// so the state can live in registers and be stored as structure-of-arrays word planes (word w of arena a at
// w * num_arenas + a). Internal units are Bullet units; the host API converts.
#pragma once

#include "math.h"

namespace rlmax {

// Car flag bits (CarState::flags)
enum CarFlag : uint32_t {
    CF_HAS_JUMPED = 1u << 0,
    CF_HAS_DOUBLE_JUMPED = 1u << 1,
    CF_HAS_FLIPPED = 1u << 2,
    CF_IS_FLIPPING = 1u << 3,
    CF_IS_JUMPING = 1u << 4,
    CF_JUMP_HOLD_BROKEN = 1u << 5,
    CF_IS_AUTO_FLIPPING = 1u << 6,
    CF_IS_SUPERSONIC = 1u << 7,
    CF_IS_BOOSTING = 1u << 8,
    CF_IS_DEMOED = 1u << 9,
    CF_ON_GROUND = 1u << 10,
    CF_STICKY_GATE = 1u << 11,        // a wheel touched the world last tick (sticky force / auto-roll gate)
    CF_WORLD_CONTACT = 1u << 12,      // world_contact_normal is valid (set in the step, consumed next pre-tick)
    CF_PREV_JUMP = 1u << 13,          // previous tick's jump input
    CF_HAS_EXTRA_HIT = 1u << 14,      // last_extra_hit_tick is valid
    CF_JUMP_GROUND = 1u << 25,        // three wheels or more touched at last tick's start: a jump re-arms on the change
    CF_AIR_JUMP_USED = 1u << 24,      // the air jump went since the car last stood on its wheels; survives a demolition
    CF_HAS_BUMP_VICTIM = 1u << 15,    // bump_victim is valid (else the cooldown applies to every victim)
    CF_WHEEL0 = 1u << 16,             // wheels 0..3 in contact (bits 16..19), from the last vehicle update
    CF_WORLD_WHEEL0 = 1u << 20,       // wheels 0..3 in contact with the world (bits 20..23), from the last vehicle update
};

// Control bits (Controls::buttons)
// BTN_JUMP_HELD: the jump button is down while the game drives no controls (kickoff countdown, goal replay). The game zeroes the car's
// controls but keeps the raw input as the last controls, so the press edge when driving resumes needs it.
// BTN_UNDRIVEN: the same time, for the drive: the throttle stays but the engine gets none and nothing brakes.
enum Button : uint32_t { BTN_JUMP = 1u << 0, BTN_BOOST = 1u << 1, BTN_HANDBRAKE = 1u << 2, BTN_JUMP_HELD = 1u << 3, BTN_UNDRIVEN = 1u << 4 };

struct Controls {
    float throttle, steer, pitch, yaw, roll;
    uint32_t buttons;
};

// One manifold point (Bullet's btManifoldPoint, the fields the simulation reads): the point on body A in A's frame,
// the point on body B in B's frame (world for the static world), the normal on B (world), the distance at the last
// refresh. A manifold holds up to 4 in Bullet's compact cache order (manifold.h).
struct MPoint {
    V3 la, lb, n;
    float d;
};
constexpr int MANIFOLD_CAPACITY = 4;
struct Manifold {
    MPoint p[MANIFOLD_CAPACITY];
    uint32_t n;
};

// A body's manifolds against the static world: one per (body, mesh component or plane) pair, as Bullet keeps one per
// collision-object pair, tagged with the static object (mesh index, or 0x100 + plane). Kept compact in creation order;
// a manifold left empty is released and the last takes its place (btCollisionDispatcher::releaseManifold). Bullet
// releases a manifold when the pair's bounding boxes stop overlapping; an empty manifold has no observable effect but
// its place in the order, so releasing it when empty differs only there.
template <int M>
struct WorldManifolds {
    Manifold m[M];
    uint32_t obj[M];
    uint32_t n;
};
constexpr int CAR_WORLD_MANIFOLDS = 4;
constexpr int BALL_WORLD_MANIFOLDS = 4;

struct CarState {
    V3 pos;  // BT
    Q rot;
    V3 vel;  // BT/s
    V3 ang_vel;
    Controls ctl;  // latched controls for this tick
    uint32_t flags;
    float boost;  // 0..100
    float flip_torque_x, flip_torque_y;  // flip_rel_torque (z is always 0)
    uint32_t jump_ticks;
    float flip_time;
    float air_time;
    float time_since_jump;  // seconds since the last jump, counted on the ground too
    float handbrake_val;
    float supersonic_timer;
    float time_since_boosted;
    float boosting_time;
    float auto_flip_timer;
    float auto_flip_torque_scale;
    float bump_cooldown;
    uint32_t bump_victim;
    uint32_t last_extra_hit_tick;
    float demo_respawn_timer;
    V3 wheel_normal_sum;      // sum of last tick's world-contact wheel normals (sticky force direction)
    V3 world_contact_normal;  // car-world contact normal from the last step
    uint32_t team;            // 0 blue, 1 orange
    uint32_t body;            // body preset (consts.h BodyPresetId)
    WorldManifolds<CAR_WORLD_MANIFOLDS> wman;  // A = car, B = static world
    uint32_t pad_touch;       // boost pad the car is touching (0xFFFFFFFF: none)
    uint32_t pad_frame_valid; // replaces the obsolete touch-arrival tick
    V3 pad_frame_pos; // last render-frame end in BT
};

struct BallState {
    V3 pos;
    Q rot;
    V3 vel;
    V3 ang_vel;
    float deactivation_time;
    uint32_t sleeping;
    M3 sleep_inv_i;  // inverse inertia held while the ball sleeps
};

template <int NC> struct SweepKeyType { using type = uint8_t; };
template <> struct SweepKeyType<8> { using type = uint16_t; };
template <int NC> using SweepKey = typename SweepKeyType<NC>::type;

template <int NC>
constexpr int sweep_cap() {
    // New counts reserve every dynamic-static and dynamic-dynamic pair. Existing counts keep their original capacity.
    if constexpr (NC == 3 || NC == 5 || NC == 7 || NC == 8)
        return (NC + 1) * 22 + NC * (NC + 1) / 2;
    else return 12 + 4 * NC;
}

template <int NC> RLMAX_HD constexpr int sweep_man_cap() { return sweep_cap<NC>() + 1 + NC; }

template <int NC>
struct SweepState {  // the broadphase sweep's state (sweep.h)
    uint32_t hdr;               // bit 0: valid; bit 1: box valid; bit 2: predictive manifolds; bits 8..15 pairs; bits 16..23 manifolds
    uint32_t predictive_visit;  // object order when the predictive manifolds were appended
    uint32_t edge[1 + NC][3];   // per body and axis: min edge position | max edge position << 16
    float box[1 + NC][6];       // per body, the proxy box the last sweep took (min xyz, max xyz): the wheel rays skip a body whose box misses theirs
    uint8_t slot[1 + NC][8];    // per body: place of the min and max edge in each axis (axis * 2 + max)
    SweepKey<NC> pairs[sweep_cap<NC>()];  // overlapping pairs in pair-cache order, as sw_key encodes them
    SweepKey<NC> man[sweep_man_cap<NC>()];    // manifolds in dispatcher order, same encoding
    uint8_t handle[1 + NC];          // per body, its order among the game's broadphase handles; 0: none (a demolished car, or a new arena)
    uint8_t free_handles[1 + NC];    // the handles released, last in first out
    uint8_t free_n;
};

constexpr int NUM_PADS = 34;

// The car visited k-th (the game's vehicle order: pre-ticks, vehicle updates, pickups). Slot k of ArenaState::visit holds
// slot ^ k, so a state that was never reordered is zero and the usual order, ascending, costs one xor.
// Seven and eight cars use three bits per slot to leave the high byte for the absent mask (seven nibbles would reach bit 27).
template <int NC> RLMAX_HD constexpr int visit_bits() { return NC >= 7 ? 3 : 4; }
template <int NC> RLMAX_HD int visit_slot(uint32_t visit, int k) {
    constexpr int bits = visit_bits<NC>();
    return (int)(((visit >> (bits * k)) & ((1u << bits) - 1)) ^ (uint32_t)k);
}

template <int NC>
struct ArenaState {
    BallState ball;
    CarState cars[NC];
    uint32_t pad_gave_tick[NUM_PADS];  // tick the pad last gave boost (0xFFFFFFFF: never)
    uint16_t pad_pickup_frac[NUM_PADS];
    uint32_t pad_frame_bits; // float bits XOR 0.5 bits; zero is the default
    Manifold pair_man[NC * (NC - 1) / 2 > 0 ? NC * (NC - 1) / 2 : 1];  // car pairs (i < j): A = car j, B = car i (body0 is the lower broadphase handle)
    Manifold ball_man[NC];                                            // A = ball, B = car i
    WorldManifolds<BALL_WORLD_MANIFOLDS> ball_wman;                   // A = ball, B = static world
    uint32_t tick;
    uint32_t rng;
    uint32_t visit;   // the order the game visits the cars in (visit_slot); bit 24 + i: car i is out of the order (demolished)
    uint32_t events;  // bit 0 blue scored, bit 1 orange scored; cleared by the host per decision
    uint32_t fault;   // non-finite state detected (DESIGN §14)
    uint32_t seed_mask;  // pending basis and inverse-valid: 9 bits each
    uint32_t countdown;  // the round countdown (countdown.h): bits 0..1 phase (0 none, 1 first second, 2 three fires), bits 2..3 fires so far
    float countdown_count;  // the countdown's seconds counter; zero when no countdown runs
    SweepState<NC> sweep;  // the order of the solver's manifolds (sweep.h)
    // Cold upload data, carried by arena words; body 0 is the ball, then the cars.
    M3 seed_basis[NC + 1];
    M3 seed_inv[NC + 1];
};

template <int NC>
constexpr int state_words() {
    return (int)(sizeof(ArenaState<NC>) / 4);
}

static_assert(sizeof(CarState) % 4 == 0, "CarState must be whole words");
static_assert(sizeof(BallState) % 4 == 0, "BallState must be whole words");

}  // namespace rlmax
