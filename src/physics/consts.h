// RLMAX constants.
// Internal units are Bullet units (1 BT = 50 uu), as in the game's Bullet world.
#pragma once

#include "math.h"

namespace rlmax {
namespace c {

constexpr float BT_TO_UU = 50.0f;
constexpr float UU_TO_BT = 1.0f / 50.0f;
constexpr float TICK_RATE = 120.0f;
constexpr float DT = 1.0f / 120.0f;
constexpr float DEG_TO_RAD = 0.01745329238474369f;
// Unreal rotator units: 65536 to a turn. The game scales an angle to them as angle * 32768 / pi, with pi as a float; its
// sine table has 16384 entries (entry j is sin(2 pi j / 16384), evaluated as ((j + j) * pi) * (1 / 16384)).
constexpr float PI_F32 = 3.14159274f;
constexpr float UE3_HALF_TURN = 32768.0f;
constexpr float UE3_TABLE_INV_SIZE = 6.103515625e-05f;
constexpr float UE3_UNIT_TO_RAD = 2.0f * 3.141592653589793f / 65536.0f;
// A sweep shorter than this many BT is not cast
constexpr float SWEEP_MIN_MOVE_BT = 0.02f;

// sim/consts.rs
constexpr float GRAVITY_Z_UU = -650.0f;

constexpr float ARENA_FRICTION = 0.6f, ARENA_RESTITUTION = 0.3f;
constexpr float ARENA_X_UU = 4096.0f, ARENA_Y_UU = 5120.0f, ARENA_Z_UU = 2048.0f;

// car
constexpr float CAR_MASS_BT = 180.0f;
constexpr float CAR_FRICTION = 0.3f, CAR_RESTITUTION = 0.1f;
constexpr float CAR_BALL_FRICTION = 2.0f, CAR_BALL_RESTITUTION = 0.0f;
constexpr float CAR_WORLD_FRICTION = 0.3f, CAR_WORLD_RESTITUTION = 0.3f;
constexpr float CAR_CAR_FRICTION = 0.09f, CAR_CAR_RESTITUTION = 0.1f;
constexpr float CAR_MAX_SPEED_UU = 2300.0f;
constexpr float CAR_MAX_ANG_SPEED = 5.5f;

constexpr float BOOST_MAX = 100.0f;
constexpr float BOOST_USED_PER_SECOND = 33.3f;
constexpr float BOOST_MIN_TIME = 0.1f;
constexpr float BOOST_ACCEL_GROUND_UU = 2975.0f / 3.0f;
constexpr float BOOST_ACCEL_AIR_UU = 2975.0f / 3.0f;
constexpr float BOOST_SPAWN_AMOUNT = 33.3f;  // TAGame CarComponent_Boost.StartBoostAmount 0.333
constexpr float BOOST_RECHARGE_PER_SECOND = 10.0f;  // only with the recharge mutator (off in Soccar)
constexpr float BOOST_RECHARGE_DELAY = 0.25f;

constexpr float SUPERSONIC_START_SPEED_UU = 2200.0f;
constexpr float SUPERSONIC_MAINTAIN_MIN_SPEED_UU = 2100.0f;
constexpr float SUPERSONIC_MAINTAIN_MAX_TIME = 1.0f;

constexpr float DEMO_YAW_LIMIT_DEG = 45.572994f, DEMO_PITCH_LIMIT_DEG = 36.869896f;

// The game's drive and brake torques (pkg HandlingPresets / DefaultVehicleSim DriveTorque 288000, BrakeTorque 10500) and the share per
// wheel (bin f72740: DAT 141a6214c, 0.25); each is multiplied by 0.02 last (bin f6b110: DAT 141a62aac). 0.25 is a power of two, so the
// order of the 0.25 is free; the rest is the game's: ((curve * throttle) * torque) * 0.25 * 0.02, (brake * torque) * 0.25 * 0.02.
constexpr float DRIVE_TORQUE_UU = 288000.0f, BRAKE_TORQUE_UU = 10500.0f, WHEEL_TORQUE_SHARE = 0.25f;
constexpr float STOPPING_FORWARD_VEL_UU = 25.0f;
constexpr float COASTING_BRAKE_FACTOR = 0.15f;
constexpr float BRAKING_NO_THROTTLE_SPEED_THRESH = 0.01f;
constexpr float THROTTLE_DEADZONE = 0.01f;
constexpr float THROTTLE_AIR_ACCEL_UU = 200.0f / 3.0f;
constexpr float POWERSLIDE_RISE_RATE = 5.0f, POWERSLIDE_FALL_RATE = 2.0f;

constexpr float JUMP_ACCEL_UU = 4375.0f / 3.0f;
constexpr float JUMP_IMMEDIATE_FORCE_UU = 875.0f / 3.0f;
constexpr uint32_t JUMP_MIN_TICKS = 3, JUMP_MAX_TICKS = 24;
constexpr float JUMP_REARM_TIME = 0.125f;
constexpr float DOUBLEJUMP_MAX_DELAY = 1.25f;

constexpr float FLIP_Z_DAMP_120 = 0.35f, FLIP_Z_DAMP_START = 0.15f, FLIP_Z_DAMP_END = 0.21f;
constexpr float FLIP_TORQUE_TIME = 0.65f, FLIP_TORQUE_MIN_TIME = 0.41f;
constexpr float FLIP_PITCH_CANCEL_GATE_MIN_TIME = 0.041f;
constexpr float FLIP_PITCHLOCK_TIME = 1.0f, FLIP_PITCHLOCK_EXTRA_TIME = 0.3f;
constexpr float FLIP_INITIAL_VEL_SCALE = 500.0f;
constexpr float FLIP_TORQUE_X = 260.0f, FLIP_TORQUE_Y = 224.0f;
constexpr float FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE = 1.0f;
constexpr float FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE = 1.9f;
constexpr float FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE = 2.5f;
constexpr float FLIP_BACKWARD_IMPULSE_SCALE_X = 16.0f / 15.0f;

constexpr float AIR_TORQUE_P = 130.0f, AIR_TORQUE_Y = 95.0f, AIR_TORQUE_R = 400.0f;
constexpr float AIR_DAMP_P = 30.0f, AIR_DAMP_Y = 20.0f, AIR_DAMP_R = 50.0f;
constexpr float AIR_TORQUE_APPLY_SCALE = 2.0f * PI / 65536.0f * 1000.0f;

constexpr float AUTOFLIP_IMPULSE_UU = 200.0f, AUTOFLIP_ANG_VEL_DELTA = 1000.0f, AUTOFLIP_TIME = 0.4f, AUTOFLIP_TIME_THREE_WHEELS = 0.2f;
constexpr float AUTOFLIP_NORM_Z_THRESH = 0.70710678118654752f, AUTOFLIP_UP_DOT = -0.966f;
constexpr float AUTOROLL_FORCE_UU = 100.0f, AUTOROLL_TORQUE = 80.0f;

constexpr float BUMP_COOLDOWN_TIME = 0.25f;
constexpr float BUMP_YAW_LIMIT_DEG = 70.0f, BUMP_PITCH_LIMIT_DEG = 36.869896f;
// tan() of the cone limits, precomputed (exact: tan in f64, rounded to f32)
// the bump and demolition cones in degrees
constexpr float BUMP_YAW_DEG = 70.0f, BUMP_PITCH_DEG = 36.869896f;
constexpr float DEMO_YAW_DEG = 45.572994f, DEMO_PITCH_DEG = 36.869896f;

constexpr float CAR_REST_Z_UU = 17.0f, DEMO_RESPAWN_TIME = 3.0f;

// ball
// GameInfo_Soccar_SF Ball_Default sphere element radius 92 uu, through the game's shape rule in float (1.825 BT, 91.25 uu):
// the physics shim's desc radius is scale * radius + 0.025 with scale 0.02 (bin 972a20 line 132), and Bullet's sphere reads it back
// as (desc * 50 * 0.02) - 0.04 (bin b2_416000.c line 203; 0.02 DAT 141a62aac, 50 DAT 141b876c0 shapes.md, 0.04 DAT 141a61a48).
// The x50 x0.02 round trip is not the identity in float: 0x3fe99998 here, 0x3fe99999 without it.
constexpr float BALL_DESC_RADIUS_BT = 0.02f * 92.0f + 0.025f;
constexpr float BALL_RADIUS_BT = BALL_DESC_RADIUS_BT * 50.0f * 0.02f - 0.04f;
constexpr float BALL_RADIUS_UU = BALL_RADIUS_BT * BT_TO_UU;
constexpr float BALL_MASS_BT = 30.000001907348633f;  // the game's float, not 30: inverse mass and rows follow it
constexpr float BALL_REST_Z_UU = 93.15f;
constexpr float BALL_KICKOFF_Z_UU = 92.75f;
constexpr float BALL_MAX_ANG_SPEED = 6.0f;
constexpr float BALL_DRAG = 0.03f;
constexpr float BALL_FRICTION = 0.35f, BALL_RESTITUTION = 0.6f;
constexpr float BALL_MAX_SPEED_UU = 6000.0f;
constexpr float BALL_LINEAR_SLEEP_THRESH = 0.8f, BALL_ANGULAR_SLEEP_THRESH = 1.0f;  // BT units, rad/s
constexpr float HIT_FORWARD_SCALE = 0.65f, HIT_MAX_DELTA_VEL_UU = 4600.0f, HIT_Z_SCALE = 0.35f;

// goal
constexpr float GOAL_THRESHOLD_Y_UU = 5124.25f;

// bullet_vehicle
constexpr float SUSPENSION_STIFFNESS = 500.0f;
constexpr float WHEELS_DAMPING_COMPRESSION = 25.0f, WHEELS_DAMPING_RELAXATION = 40.0f;
constexpr float MAX_SUSPENSION_TRAVEL_UU = 12.0f;
constexpr float ROLLING_FRICTION_SCALE = 113.73963f;    // vehicle/wheel_info.rs

// boost pads
constexpr float PAD_CYL_HEIGHT = 95.0f, PAD_CYL_RAD_BIG = 208.0f, PAD_CYL_RAD_SMALL = 176.0f;
constexpr float PAD_BOX_HEIGHT = 64.0f, PAD_BOX_RAD_BIG = 160.0f, PAD_BOX_RAD_SMALL = 96.0f;
constexpr float PAD_COOLDOWN_BIG = 10.0f, PAD_COOLDOWN_SMALL = 4.0f;
constexpr float PAD_AMOUNT_BIG = 100.0f, PAD_AMOUNT_SMALL = 12.0f;

// solver
constexpr int SOLVER_ITERATIONS = 10;
constexpr float ERP2 = 0.8f;
constexpr float SPLIT_IMPULSE_TURN_ERP = 0.1f;
constexpr float WHEEL_PUSHBACK_ERP = 0.1f;
constexpr float CONTACT_BREAKING_THRESHOLD = 0.02f;  // BT
constexpr float ANGULAR_MOTION_THRESHOLD = PI * 0.25f;
constexpr float CONVEX_DISTANCE_MARGIN = 0.04f;

// network quantisation
constexpr float QUANT_POS_SCALE = 100.0f, QUANT_VEL_SCALE = 100.0f, QUANT_ANG_VEL_SCALE = 10000.0f;

// Collision shapes. The game builds each shape from the package's RB body
// setup in Bullet units and adds a 0.025 BT skin to every half extent and radius (bin 972a20); a sphere's Bullet radius
// is that minus Bullet's default 0.04 margin (bin 416140).
constexpr float SHAPE_SKIN_BT = 0.025f;
constexpr float BULLET_DEFAULT_MARGIN_BT = 0.04f;
// The game's sphere reports its bounds as radius + 0.04 + 0.04 (both the default margin constant): its AABB
// (getAabb, bin 2a6d40) and its bounding sphere (2a7080), hence its angular motion disc and breaking threshold.
constexpr float SPHERE_BOUND_PAD_BT = BULLET_DEFAULT_MARGIN_BT + BULLET_DEFAULT_MARGIN_BT;

}  // namespace c

// Car body presets, in the game's preset order. Box full lengths and
// offset (x, z; uu): Startup HandlingPresets Physics_<preset> RB_BodySetup box element. Wheels (radius, suspension rest
// length, connection point; uu): from the game's body configs. Steering geometry comes from the
// wheel positions (wheelbase front x - back x, front track 2 front y; bin ed88e0, ed5a00).
struct BodyPreset {
    float box_x, box_y, box_z, off_x, off_z;
    float front_rad, back_rad, front_rest, back_rest;
    float front_x, front_y, front_z, back_x, back_y, back_z;
};
constexpr int NUM_BODY_PRESETS = 8;
enum BodyPresetId : uint32_t { BODY_OCTANE, BODY_DOMINUS, BODY_PLANK, BODY_BREAKOUT, BODY_HYBRID, BODY_MERC, BODY_BATMOBILE, BODY_PSYCLOPS };

// The height a body respawns at, uu (the suite's recorded CarConfig.spawn_z, the common value where the game's cars sharing a preset differ)
RLMAX_HD float respawn_z_uu(uint32_t i) {
    constexpr float Z[NUM_BODY_PRESETS] = {36.02f, 36.01f, 36.0f, 36.0f, 37.53f, 35.05f, 36.0f, 36.0f};
    return Z[i < (uint32_t)NUM_BODY_PRESETS ? i : 0u];
}

RLMAX_HD const BodyPreset& body_preset(uint32_t i) {
    static constexpr BodyPreset P[NUM_BODY_PRESETS] = {
        // Physics_Octane
        {118.00738f, 84.19941f, 36.159073f, 13.87566f, 20.754988f, 12.5f, 15.0f, 38.754988f, 37.054988f,
         51.25f, 25.90f, 20.754988f, -33.75f, 29.50f, 20.754988f},
        // Physics_MuscleCar (Dominus)
        {127.92678f, 83.27995f, 31.3f, 9.0f, 15.75f, 12.0f, 13.5f, 33.95f, 33.85f,
         50.30f, 31.10f, 15.75f, -34.75f, 33.00f, 15.75f},
        // Physics_Plank
        {128.81978f, 84.670364f, 29.394402f, 9.008572f, 12.0942f, 12.5f, 17.0f, 31.9242f, 27.9242f,
         49.97f, 27.799904f, 12.0942f, -35.43f, 20.280005f, 12.0942f},
        // Physics_LongCar (Breakout)
        {131.49236f, 80.521f, 30.3f, 12.5f, 11.75f, 13.5f, 15.0f, 29.7f, 29.66f,
         51.50f, 26.67f, 11.75f, -35.75f, 35.00f, 11.75f},
        // Physics_Hybrid
        {127.01919f, 82.18787f, 34.159073f, 13.87566f, 20.754988f, 12.5f, 15.0f, 38.754988f, 37.054988f,
         51.25f, 25.90f, 20.754988f, -34.00f, 29.50f, 20.754988f},
        // Physics_Merc
        {120.72023f, 76.71031f, 41.659073f, 11.37566f, 21.504988f, 15.0f, 15.0f, 39.504988f, 39.104988f,
         51.25f, 25.90f, 21.504988f, -33.75f, 29.50f, 21.504988f},
        // The Batmobile: the Plank's box and rests, its own wheel y and steering
        {128.81978f, 84.670364f, 29.394402f, 9.008572f, 12.0942f, 12.5f, 17.0f, 31.9242f, 27.9242f,
         49.97f, 27.80f, 12.0942f, -35.43f, 20.28f, 12.0942f},
        // Psyclops.
        {118.00738f, 84.19941f, 36.159073f, 13.87566f, 15.00499f, 12.5f, 15.0f, 33.00499f, 31.304989f,
         51.25f, 5.0f, 15.00499f, -34.25f, 29.5f, 15.00499f},
    };
    return P[i < NUM_BODY_PRESETS ? i : 0];
}

// The steering knots after the game's init rewrite, per body preset (degrees): steer at |forward speed| 0, 500, 1000, 1500,
// 1750, 3000 uu/s, then the handbrake curve at 0, 2500. The game computes, in float and in this order (bin ed5a00, read
// 3 Nov 2026): x = knot * 0.017453292 (DAT 141a6213c); a = atanf(L / (T + L / tanf(x))); mean = (((a - x) + x) - x) * 0.5 + x;
// out = mean * 57.29578 (DAT 141c4a390); L = |front x - rear x|, T = |front-left y - front-right y| of the handling
// preset's axle rest positions (pkg HandlingPresets.Presets.*, Plank y 27.799904, not 27.80). Values computed with
// the Windows UCRT tanf / atanf (equal to the double function rounded to float at every knot); curves in radians equal them to 3e-5 rad.
struct SteerKnots {
    float steer[6], powerslide[2];
};
RLMAX_HD const SteerKnots& steer_knots_deg(uint32_t i) {
    static constexpr SteerKnots K[NUM_BODY_PRESETS] = {
        {{30.570510864257812f, 18.294261932373047f, 10.429570198059082f, 6.0558881759643555f, 4.874049186706543f, 1.9791786670684814f},
         {22.478458404541016f, 7.224311351776123f}},  // Octane
        {{29.923809051513672f, 18.018766403198242f, 10.329229354858398f, 6.019903182983398f, 4.85035514831543f, 1.9751166105270386f},
         {22.08690071105957f, 7.173928260803223f}},  // Dominus
        {{30.342710494995117f, 18.1981258392334f, 10.394840240478516f, 6.043498516082764f, 4.865903377532959f, 1.9777874946594238f},
         {22.341325759887695f, 7.206938743591309f}},  // Plank
        {{30.55976104736328f, 18.289745330810547f, 10.42794418334961f, 6.055309772491455f, 4.873669147491455f, 1.9791138172149658f},
         {22.472003936767578f, 7.2235002517700195f}},  // Breakout
        {{30.580455780029297f, 18.29843521118164f, 10.431071281433105f, 6.056422233581543f, 4.874399185180664f, 1.9792383909225464f},
         {22.484424591064453f, 7.225060939788818f}},  // Hybrid
        {{30.570510864257812f, 18.294261932373047f, 10.429570198059082f, 6.0558881759643555f, 4.874049186706543f, 1.9791786670684814f},
         {22.478458404541016f, 7.224311351776123f}},  // Merc
        {{30.342700958251953f, 18.1981201171875f, 10.394838333129883f, 6.0434980392456055f, 4.865902900695801f, 1.9777874946594238f},
         {22.341318130493164f, 7.206938743591309f}},  // Batmobile
        {{30.570510864257812f, 18.294261932373047f, 10.429570198059082f, 6.0558881759643555f, 4.874049186706543f, 1.9791786670684814f},
         {24.427291870117188f, 7.443766117095947f}},  // Psyclops
    };
    return K[i < NUM_BODY_PRESETS ? i : 0];
}

// A car's rest state at the start of play (kickoff_reset), per body: the state RLMAX's physics settles to when the car is
// dropped from the spawn height (z 36.02 uu, flat, still, in the air) and left idle for 1,200 ticks; the pose, velocities
// and angular velocity are word-identical at 2,400 ticks. The car faces +y (yaw pi/2) at x 5.12 BT; the table keeps the
// pose's z, rotation quaternion (x, y, z, w) and velocities, which kickoff_reset turns about z to each spot's yaw. The
// flags, wheel_normal_sum and world_contact_normal of the settled car are the constants set in kickoff_reset (flags
// 0x2ff0c00, normal sum (0, 0, 4), contact normal 0), the same for every body.
// Why a drop: the rest height depends on the path. The tick-start position and velocity codec (body.h quantize_body)
// holds a slowly rising car at the height it reached, so a car started at rest at z 17 stops at a different height than
// one dropped from spawn (Octane 17.003 against 17.0123). The game's cars drop from spawn: the first physics row of every
// recorded kickoff and respawn is a car still falling, and RLMAX dropped from z 36.02 reproduces that row.
// The recordings' cars at "go" (the last row before any car moves, 450 rows after the kickoff's first physics row) are
// at z 17.0123 (Octane), 18.6523 (Plank), 18.7919 (Merc); the table's settled z is 17.01226, 18.65233, 18.79193. Those
// three rows are confirmed by the recordings. Dominus (17.0527), Breakout (18.3420) and Hybrid (17.0121) are RLMAX's
// settle only: no recording has those bodies. Float literals round-trip, not rounded. Generated by an idle settle
// (spawn 36.0, 36.02 and 38.61 give the same z for all six bodies).
struct KickoffRest {
    float z;            // BT
    float vel[3];       // BT/s
    float ang[3];       // rad/s
    float rot[4];       // x, y, z, w
};
RLMAX_HD const KickoffRest& kickoff_rest(uint32_t i) {
    static constexpr KickoffRest R[NUM_BODY_PRESETS] = {
        // Physics_Octane (recordings)
        {0.34024516f, {9.626798e-9f, 0.0002042306f, 0.0054183453f}, {-0.0006167898f, 1.8626166e-7f, -1.3919816e-9f}, {-0.003376262f, 0.003376263f, 0.707096f, 0.70710146f}},
        // Physics_MuscleCar, Dominus (RLMAX settle)
        {0.3410536f, {8.579142e-9f, 0.00024794205f, 0.0064348727f}, {-0.00072966435f, 1.4900762e-7f, -1.8121682e-9f}, {-0.00591219f, 0.0059121493f, 0.70708436f, 0.70707977f}},
        // Physics_Plank (recordings)
        {0.37304667f, {-2.331186e-9f, -0.00024040767f, 0.005602181f}, {0.00051143474f, -7.447653e-9f, 4.3655746e-11f}, {-0.0021267403f, 0.0021267403f, 0.70710135f, 0.7071058f}},
        // Physics_LongCar, Breakout (RLMAX settle)
        {0.3668402f, {-2.5584145e-6f, 0.00022473664f, 0.004829213f}, {-0.000608454f, -4.555136e-5f, 6.142242e-7f}, {-0.0060144705f, 0.006019555f, 0.7070745f, 0.70708793f}},
        // Physics_Hybrid (RLMAX settle)
        {0.34024197f, {-4.95618e-9f, 0.0002042306f, 0.005034104f}, {-0.0006202181f, -1.0430658e-7f, 8.362804e-10f}, {-0.0033654831f, 0.0033654827f, 0.7071015f, 0.707096f}},
        // Physics_Merc (recordings)
        {0.37583858f, {-1.88895e-6f, 0.00018418208f, 0.004632652f}, {-0.00052526104f, -3.4787277e-5f, 2.0540938e-9f}, {0.0017165927f, -0.0017114027f, 0.7071081f, 0.70710135f}},
        // the Batmobile (the Plank's row for now: same box and rests, wheel y differs by 1e-4 uu)
        {0.37304667f, {-2.331186e-9f, -0.00024040767f, 0.005602181f}, {0.00051143474f, -7.447653e-9f, 4.3655746e-11f}, {-0.0021267403f, 0.0021267403f, 0.70710135f, 0.7071058f}},
        // Psyclops: CPU drop from 36 uu, identical at 1,200 and 2,400 ticks; not game-confirmed.
        {0.3532387f, {-3.3650895e-7f, -0.00018300959f, 0.0046462417f}, {0.0004241059f, -2.1233996e-5f, 1.3072713e-7f}, {-0.0033624691f, 0.0033677383f, 0.707096f, 0.70710164f}},
    };
    return R[i < NUM_BODY_PRESETS ? i : 0];
}

// Full half extents of a preset's Bullet box, in the game's float order: the shim desc is scale * length * 0.5 + skin (bin 972a20:232), and
// Bullet's box reads it back as (desc * 50) * 0.02 (bin b2_416000.c case 2, lines 214-215). The round trip moves one ulp for Dominus y and
// Hybrid x, z (census, 3 Nov 2026).
RLMAX_HD V3 body_half_bt(const BodyPreset& p) {
    const float hx = 0.02f * p.box_x * 0.5f + c::SHAPE_SKIN_BT, hy = 0.02f * p.box_y * 0.5f + c::SHAPE_SKIN_BT,
                hz = 0.02f * p.box_z * 0.5f + c::SHAPE_SKIN_BT;
    return v3(hx * 50.0f * 0.02f, hy * 50.0f * 0.02f, hz * 50.0f * 0.02f);
}

// Behaviours RLMAX reproduces. Each may be an inaccuracy of the model it was fitted to rather than
// the game's rule, so each is a named switch: one can be turned off on evidence without touching the
// kernel's structure.
namespace quirk {
// Network-style quantisation of position (0.01 uu), velocity (0.01 uu/s) and angular velocity each tick.
constexpr bool QUANTIZE_PHYS = true;
// Ball-vs-static contacts become split-only rows plus one averaged synthetic row per body.
constexpr bool BALL_STATIC_AVERAGED_ROW = true;
// Velocity iterations stop once the maximum squared residual is exactly zero.
constexpr bool SOLVER_EARLY_EXIT = true;
// Box vs plane emits the four nearest corners per tick instead of one support point (off: the plane pair's persistent
// manifold builds its points up over ticks, CAR_WORLD_PERSISTENT).
constexpr bool BOX_PLANE_FOUR_CORNERS = false;
// Car-world manifolds persist across ticks, one per (car, mesh component or plane) pair, as Bullet's do (manifold.h).
constexpr bool CAR_WORLD_PERSISTENT = true;
// A car's pairs refresh their manifold before adding this tick's points (btCompoundCollisionAlgorithm refreshes its
// child manifolds first; the child algorithm refreshes again after adding).
constexpr bool COMPOUND_REFRESH_FIRST = true;
// A car's box meets only the mesh triangles that touch its bounding box (btConvexTriangleCallback::processTriangle).
constexpr bool TRI_AABB_GATE_CAR = true;
// The car-ball pair runs, and keeps its manifold, only while the box's and the ball's bounding boxes overlap
// (btCompoundCollisionAlgorithm's child AABB test).
constexpr bool CHILD_AABB_GATE_BALL = true;
// The ball meets only the mesh triangles that touch its bounding box (the same callback as the car).
constexpr bool TRI_AABB_GATE_BALL = true;
// Box-triangle distance by Bullet's GJK (btGjkPairDetector, Voronoi simplex solver) instead of exact closest features.
constexpr bool GJK_BOX_TRI = true;
// Box-triangle penetration by Bullet's GJK-EPA (btGjkEpaSolver2) instead of the SAT least-penetration axis.
constexpr bool EPA_BOX_TRI = true;
// Skip the box-triangle GJK when the implicit box's separation along the triangle normal already rules out a contact
// (the same outcome as running it; a speed-only shortcut).
constexpr bool GJK_PLANE_REJECT = true;
// The vehicle update runs inside the Bullet step after gravity, the speed clamp and damping, once every car's
// components have run (bin 2b8fc0 step d), instead of at the end of each car's pre-tick as the earlier model ran it.
constexpr bool VEHICLE_IN_STEP = true;
// Wheel rays reject hits closer than sqrt(0.116684) BT to the hard point.
constexpr bool WHEEL_RAY_MIN_HIT_DIST = true;
// Sticky force and auto-roll are gated on the previous tick's world wheel contact.
constexpr bool STICKY_GATE_PREV_TICK = true;
}  // namespace quirk

// Piecewise-linear curve, clamped at the ends (sim/linear_piece_curve.rs semantics).
template <int N>
struct Curve {
    float x[N], y[N];
    RLMAX_HD float eval(float in) const {
        if (in <= x[0]) return y[0];
        for (int i = 1; i < N; i++) {
            if (in < x[i]) {
                float t = (in - x[i - 1]) / (x[i] - x[i - 1]);
                return y[i - 1] + (y[i] - y[i - 1]) * t;
            }
        }
        return y[N - 1];
    }
};

namespace curves {
// Steering, degrees vs |forward speed| (GameInfo_Soccar_SF Car_Default DefaultVehicleSim SteerAngleCurve and
// HandbrakeSteerAngleCurve). The game rewrites every knot at vehicle init (bin f282e0 -> ed5a00, shapes.md S3); the
// rewritten knots are the per-body table steer_knots_deg() below, not these.
RLMAX_CONST Curve<6> STEER_ANGLE_DEG{{0.f, 500.f, 1000.f, 1500.f, 1750.f, 3000.f}, {35.f, 20.f, 11.f, 6.25f, 5.f, 2.f}};
RLMAX_CONST Curve<2> POWERSLIDE_STEER_ANGLE_DEG{{0.f, 2500.f}, {25.f, 7.5f}};
RLMAX_CONST Curve<5> DRIVE_SPEED_TORQUE_FACTOR{{-1410.f, -1400.f, 0.f, 1400.f, 1410.f}, {0.0f, 0.1f, 1.0f, 0.1f, 0.0f}};
RLMAX_CONST Curve<3> NON_STICKY_FRICTION_FACTOR{{0.f, 0.707f, 1.f}, {0.1f, 0.5f, 1.0f}};
RLMAX_CONST Curve<2> LAT_FRICTION{{0.f, 1.f}, {1.0f, 0.2f}};
RLMAX_CONST Curve<2> PSYCLOPS_LAT_FRICTION{{0.f, 1.f}, {0.30f, 0.25f}};
RLMAX_CONST Curve<1> HANDBRAKE_LAT_FRICTION_FACTOR{{0.f}, {0.1f}};
RLMAX_CONST Curve<2> HANDBRAKE_LONG_FRICTION_FACTOR{{0.f, 1.f}, {0.5f, 0.9f}};
RLMAX_CONST Curve<5> BALL_CAR_EXTRA_IMPULSE_FACTOR{{0.f, 500.f, 1400.f, 2300.f, 4600.f},
                                                 {0.65f, 0.65f, 0.60f, 0.55f, 0.30f}};
}  // namespace curves

}  // namespace rlmax
