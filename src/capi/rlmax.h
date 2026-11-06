/* RLMAX C ABI, version 1. Public state format (DESIGN §10): plain structs in Unreal units, rotations as a 3x3 matrix
 * whose columns are the body's forward, right and up axes. The internal arena layout is private to the library and
 * reached only through these calls. */
#ifndef RLMAX_CAPI_H
#define RLMAX_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RLMAX_ABI_VERSION 2u

typedef struct {
    float throttle, steer, pitch, yaw, roll;
    uint32_t jump, boost, handbrake;
} RlmaxControls;

typedef struct {
    float pos[3];
    float rot[9]; /* columns: forward (0..2), right (3..5), up (6..8) */
    float vel[3];
    float ang_vel[3];
} RlmaxBody;

typedef struct {
    RlmaxBody body;
    uint32_t sleeping; /* 0 active, 1 wants deactivation, 2 sleeping */
    float deactivation_time;
} RlmaxBall;

/* Car flag bits (same values as the internal CarFlag enum) */
#define RLMAX_CAR_HAS_JUMPED (1u << 0)
#define RLMAX_CAR_HAS_DOUBLE_JUMPED (1u << 1)
#define RLMAX_CAR_HAS_FLIPPED (1u << 2)
#define RLMAX_CAR_IS_FLIPPING (1u << 3)
#define RLMAX_CAR_IS_JUMPING (1u << 4)
#define RLMAX_CAR_JUMP_HOLD_BROKEN (1u << 5)
#define RLMAX_CAR_IS_AUTO_FLIPPING (1u << 6)
#define RLMAX_CAR_IS_SUPERSONIC (1u << 7)
#define RLMAX_CAR_IS_BOOSTING (1u << 8)
#define RLMAX_CAR_IS_DEMOED (1u << 9)
#define RLMAX_CAR_ON_GROUND (1u << 10)
#define RLMAX_CAR_STICKY_GATE (1u << 11)
#define RLMAX_CAR_WORLD_CONTACT (1u << 12)
#define RLMAX_CAR_PREV_JUMP (1u << 13)
#define RLMAX_CAR_HAS_EXTRA_HIT (1u << 14)
#define RLMAX_CAR_HAS_BUMP_VICTIM (1u << 15)
#define RLMAX_CAR_WHEEL0 (1u << 16) /* wheels 0..3 in contact: bits 16..19 */

typedef struct {
    RlmaxBody body;
    RlmaxControls controls;
    uint32_t flags;
    uint32_t team;
    uint32_t preset; /* body preset: 0 Octane, 1 Dominus, 2 Plank, 3 Breakout, 4 Hybrid, 5 Merc */
    float boost; /* 0..100 */
    float flip_torque[2];
    uint32_t jump_ticks;
    float flip_time, air_time, time_since_jump, handbrake_val, supersonic_timer;
    float time_since_boosted, boosting_time, auto_flip_timer, auto_flip_torque_scale;
    float bump_cooldown;
    uint32_t bump_victim;
    uint32_t last_extra_hit_tick;
    float demo_respawn_timer;
    float wheel_normal_sum[3];
    float world_contact_normal[3];
} RlmaxCar;

/* Static arena data built by the host (see crates/rlmax/src/arena.rs). Pointers stay owned by the caller. */
typedef struct {
    const void* tris; /* 64-byte triangles */
    const uint32_t* cell_off;
    const uint16_t* cell_idx;
    const float* goal_ly; /* goal triangles' local-frame vertex y, three per triangle */
    const float* tri_box; /* each triangle's BVH leaf box in its body's frame: min, max */
    float grid_min[3];
    float inv_cell;
    int32_t nx, ny, nz;
    uint32_t max_cell_count;
} RlmaxArenaData;

uint32_t rlmax_abi_version(void);
uint32_t rlmax_state_words(uint32_t num_cars);
uint32_t rlmax_staged_words(uint32_t num_cars);

/* Arena state is `rlmax_state_words(num_cars)` 32-bit words. `init` writes an empty arena: ball at rest at the centre,
 * cars parked out of the way with team = car index % 2. */
int rlmax_init_arena(uint32_t num_cars, uint32_t* state);
/* Reset to a kickoff: the arena's rng is set to `seed`, then the five spots are shuffled. */
int rlmax_kickoff(uint32_t num_cars, uint32_t* state, uint32_t seed);
// The trainer's extract (src/physics/extract.h): floats per arena row, one arena's row, the field names of the arena
// block (which 0; pad_0..pad_33 follow it) and of a car block (which 1), comma-separated with a trailing comma.
uint32_t rlmax_extract_row_floats(uint32_t num_cars);
int rlmax_extract(uint32_t num_cars, const uint32_t* state, float* out);
const char* rlmax_extract_fields(int which);
int rlmax_get_ball(uint32_t num_cars, const uint32_t* state, RlmaxBall* out);
int rlmax_set_ball(uint32_t num_cars, uint32_t* state, const RlmaxBall* in);
int rlmax_get_car(uint32_t num_cars, const uint32_t* state, uint32_t car, RlmaxCar* out);
int rlmax_set_car(uint32_t num_cars, uint32_t* state, uint32_t car, const RlmaxCar* in);
/* Preserve a basis and optional inverse inertia (rows), body = car index or num_cars for ball. */
int rlmax_set_body_basis(uint32_t num_cars, uint32_t* state, uint32_t body, const float basis[9], const float* inverse);
/* Fraction of the current render frame; negative means an earlier tick of that frame. */
int rlmax_set_pad_frame_fraction(uint32_t num_cars, uint32_t* state, float fraction);
int rlmax_set_controls(uint32_t num_cars, uint32_t* state, uint32_t car, const RlmaxControls* in);
uint32_t rlmax_tick_count(uint32_t num_cars, const uint32_t* state);
uint32_t rlmax_events(uint32_t num_cars, uint32_t* state, int clear);
/* Which pair kinds hold a contact point after the last tick (their persistent manifold is not empty): bit 0 any car
 * with the ball, bit 1 any pair of cars. For characterising workloads, not the physics. */
uint32_t rlmax_contact_census(uint32_t num_cars, const uint32_t* state);
/* The tick's whitelisted local-memory arrays, bytes per thread (P2 gate, SASS clause): out[0] EPA scratch, out[1] the
 * solver, out[2] the contact points, out[3] the per-body arrays (bodies, car geometry, car tick outputs, snapshots). */
int rlmax_local_sizes(uint32_t num_cars, uint32_t out[4]);

/* Refresh a car's previous-tick wheel gate (sticky force and auto-roll) from its current pose without advancing time:
 * casts the four wheel rays and sets RLMAX_CAR_STICKY_GATE and wheel_normal_sum. Place every body first. */
int rlmax_refresh_sticky_gate(uint32_t num_cars, uint32_t* state, uint32_t car, const RlmaxArenaData* arena);

/* Only wheels marked as touching in the previous-step mask (RLMAX wheel order) can contribute. */
int rlmax_refresh_sticky_gate_with_wheels(uint32_t num_cars, uint32_t* state, uint32_t car, uint32_t wheels, const RlmaxArenaData* arena);

/* Seed a car's sticky gate and wheel normals from a recorded state's previous-step wheel contacts, for a
 * set state: wheel_mask bit w = wheel w touched, bit 5 = they are the game's own world contacts, kept as they are;
 * normals = 4 x 3 floats of their contact normals (RLMAX's wheel order).
 * Keeps the wheels that touched the world (not the ball or another car); sets RLMAX_CAR_STICKY_GATE, the world-wheel
 * bits 20..23 and wheel_normal_sum. Place every body first. */
int rlmax_seed_wheel_contacts(uint32_t num_cars, uint32_t* state, uint32_t car, uint32_t wheel_mask, const float* normals,
                              const RlmaxArenaData* arena);

/* CPU build: step `num_arenas` contiguous arena states (array of structures) by `ticks` ticks. */
int rlmax_cpu_step(uint32_t num_cars, uint32_t* states, uint32_t num_arenas, uint32_t ticks, const RlmaxArenaData* arena);

#ifdef __cplusplus
}
#endif

#endif
