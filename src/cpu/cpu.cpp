// RLMAX CPU build: the same physics headers as the GPU, compiled for the host. In P2 it is the kernel's oracle and the
// RLPR backend; in P3 it becomes the SIMD build (DESIGN §3.9).
#include <string.h>

#include <type_traits>

#include "../capi/rlmax.h"
#include "../physics/tick.h"
#include "../physics/extract.h"

using namespace rlmax;

namespace {

V3 v3_from(const float* p) { return v3(p[0], p[1], p[2]); }
void v3_to(V3 v, float* p) { p[0] = v.x; p[1] = v.y; p[2] = v.z; }

M3 m3_from(const float* r) { return M3{v3(r[0], r[1], r[2]), v3(r[3], r[4], r[5]), v3(r[6], r[7], r[8])}; }
template <int NC>
void seed_body(ArenaState<NC>& arena, uint32_t body, const float* basis, const float* inverse) {
    const uint32_t bit = 1u << body;
    arena.seed_basis[body] = m3_from(basis);
    if (inverse) arena.seed_inv[body] = m3_from(inverse);
    arena.seed_mask = (arena.seed_mask & ~(bit << 20)) | bit | (inverse ? bit << 20 : 0u);
}

template <int NC>
bool seed_current(const ArenaState<NC>& arena, uint32_t body) {
    const uint32_t bit = 1u << body;
    return (arena.seed_mask & bit) != 0;
}

void m3_to(const M3& m, float* r) {
    v3_to(m.c0, r);
    v3_to(m.c1, r + 3);
    v3_to(m.c2, r + 6);
}

// uu in, BT inside: divided by 50, not multiplied by 1/50 (0.02f is inexact), so a recorded value (the game's BT times
// 50) maps back to the BT whose product is that value; the multiply lands one tick-start position quantum off on ~0.2%
// of coordinates (a tie at x * 100), which a free run carries for the whole segment
void body_in(const RlmaxBody& b, V3* pos, Q* rot, V3* vel, V3* ang) {
    *pos = v3_from(b.pos) / c::BT_TO_UU;
    *rot = q_normalize(bullet_get_rotation(m3_from(b.rot)));  // the game's quaternion of its basis (getRotation)
    *vel = v3_from(b.vel) / c::BT_TO_UU;
    *ang = v3_from(b.ang_vel);
}
void body_out(V3 pos, Q rot, V3 vel, V3 ang, RlmaxBody* b) {
    v3_to(pos * c::BT_TO_UU, b->pos);
    m3_to(bullet_set_rotation(rot), b->rot);  // the basis the game holds after integration (setRotation)
    v3_to(vel * c::BT_TO_UU, b->vel);
    v3_to(ang, b->ang_vel);
}

Controls controls_in(const RlmaxControls& c) {
    Controls o;
    o.throttle = c.throttle;
    o.steer = c.steer;
    o.pitch = c.pitch;
    o.yaw = c.yaw;
    o.roll = c.roll;
    o.buttons = ((c.jump & 1u) ? BTN_JUMP : 0u) | ((c.jump & 2u) ? BTN_JUMP_HELD : 0u) | ((c.jump & 4u) ? BTN_UNDRIVEN : 0u) | (c.boost ? BTN_BOOST : 0u) | (c.handbrake ? BTN_HANDBRAKE : 0u);
    return o;
}
RlmaxControls controls_out(const Controls& c) {
    RlmaxControls o;
    o.throttle = c.throttle;
    o.steer = c.steer;
    o.pitch = c.pitch;
    o.yaw = c.yaw;
    o.roll = c.roll;
    o.jump = (c.buttons & BTN_JUMP) ? 1u : 0u;
    o.boost = (c.buttons & BTN_BOOST) ? 1u : 0u;
    o.handbrake = (c.buttons & BTN_HANDBRAKE) ? 1u : 0u;
    return o;
}

// EPA's scratch for the CPU build: one per thread
EpaScratch& cpu_scratch() {
    static thread_local EpaScratch s;
    return s;
}

template <int NC>
int dispatch_step(uint32_t* states, uint32_t n, uint32_t ticks, const ArenaData& ad) {
    ArenaState<NC>* a = reinterpret_cast<ArenaState<NC>*>(states);
    for (uint32_t i = 0; i < n; i++) {
        GeoCache<NC> gc;  // the cars' geometry, kept across this arena's ticks (tick.h)
        gc.reset();
        for (uint32_t t = 0; t < ticks; t++) {
#ifdef RLMAX_IMPULSE_LOG
            imp_clear();  // the log holds the last tick stepped
#endif
            tick<NC>(a[i], ad, cpu_scratch(), gc);
        }
    }
    return 0;
}

template <typename F>
int with_nc(uint32_t nc, F&& f) {
    switch (nc) {
        case 2: return f(std::integral_constant<int, 2>{});
        case 3: return f(std::integral_constant<int, 3>{});
        case 4: return f(std::integral_constant<int, 4>{});
        case 5: return f(std::integral_constant<int, 5>{});
        case 6: return f(std::integral_constant<int, 6>{});
        case 8: return f(std::integral_constant<int, 8>{});
        default: return -1;
    }
}

}  // namespace

extern "C" {

uint32_t rlmax_abi_version(void) { return RLMAX_ABI_VERSION; }

uint32_t rlmax_state_words(uint32_t nc) {
    switch (nc) {
        case 2: return (uint32_t)state_words<2>();
        case 3: return (uint32_t)state_words<3>();
        case 4: return (uint32_t)state_words<4>();
        case 5: return (uint32_t)state_words<5>();
        case 6: return (uint32_t)state_words<6>();
        case 8: return (uint32_t)state_words<8>();
        default: return 0;
    }
}

int rlmax_init_arena(uint32_t nc, uint32_t* state) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto* a = reinterpret_cast<ArenaState<NC>*>(state);
        memset(a, 0, sizeof(*a));
        a->ball.pos = v3(0, 0, c::BALL_REST_Z_UU * c::UU_TO_BT);
        a->ball.rot = q_identity();
        for (int i = 0; i < NC; i++) {
            CarState& cs = a->cars[i];
            cs.team = (uint32_t)(i % 2);
            cs.pos = v3(-3000.0f + 400.0f * (float)i, cs.team ? 4000.0f : -4000.0f, c::CAR_REST_Z_UU) * c::UU_TO_BT;
            cs.rot = q_identity();
            cs.boost = c::BOOST_SPAWN_AMOUNT;
            cs.flags = CF_ON_GROUND;
        }
        for (int i = 0; i < NC; i++) a->cars[i].pad_touch = PAD_NONE;
        for (int p = 0; p < NUM_PADS; p++) a->pad_gave_tick[p] = PAD_NONE;
        a->rng = 0x9E3779B9u;
        return 0;
    });
}

// The trainer's extract (physics/extract.h): floats per arena row, one arena's row, and the field names of the arena
// block (which 0; the pad timers pad_0..pad_33 follow it) and of a car block (which 1), comma-separated.
uint32_t rlmax_extract_row_floats(uint32_t nc) {
    switch (nc) {
        case 2: return (uint32_t)extract_row_floats<2>();
        case 3: return (uint32_t)extract_row_floats<3>();
        case 4: return (uint32_t)extract_row_floats<4>();
        case 5: return (uint32_t)extract_row_floats<5>();
        case 6: return (uint32_t)extract_row_floats<6>();
        case 8: return (uint32_t)extract_row_floats<8>();
        default: return 0;
    }
}

int rlmax_extract(uint32_t nc, const uint32_t* state, float* out) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        extract_arena<NC>(*reinterpret_cast<const ArenaState<NC>*>(state), out);
        return 0;
    });
}

const char* rlmax_extract_fields(int which) {
#define RLMAX_EXTRACT_NAME(f) #f ","
    static const char arena[] = RLMAX_EXTRACT_ARENA_FIELDS(RLMAX_EXTRACT_NAME);
    static const char car[] = RLMAX_EXTRACT_CAR_FIELDS(RLMAX_EXTRACT_NAME);
#undef RLMAX_EXTRACT_NAME
    return which == 0 ? arena : which == 1 ? car : "";
}

int rlmax_kickoff(uint32_t nc, uint32_t* state, uint32_t seed) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto* a = reinterpret_cast<ArenaState<NC>*>(state);
        a->rng = seed;
        kickoff_reset<NC>(*a);
        a->sweep.hdr = 0;  // the broadphase sweep is rebuilt from the new poses
        return 0;
    });
}

int rlmax_get_ball(uint32_t nc, const uint32_t* state, RlmaxBall* out) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        const auto* a = reinterpret_cast<const ArenaState<NC>*>(state);
        body_out(a->ball.pos, a->ball.rot, a->ball.vel, a->ball.ang_vel, &out->body);
        if (seed_current(*a, 0)) m3_to(a->seed_basis[0], out->body.rot);
        out->sleeping = a->ball.sleeping;
        out->deactivation_time = a->ball.deactivation_time;
        return 0;
    });
}

int rlmax_set_ball(uint32_t nc, uint32_t* state, const RlmaxBall* in) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto* a = reinterpret_cast<ArenaState<NC>*>(state);
        body_in(in->body, &a->ball.pos, &a->ball.rot, &a->ball.vel, &a->ball.ang_vel);
        seed_body(*a, 0, in->body.rot, nullptr);
        a->ball.sleeping = in->sleeping;
        a->ball.deactivation_time = in->deactivation_time;
        a->ball.sleep_inv_i = world_inv_inertia(bullet_set_rotation(a->ball.rot), tick_consts().ball_inv_inertia);
        return 0;
    });
}

// A body's position and velocity as the state holds them, in Bullet units (cars 0 to nc - 1, then the ball): the game's own float32 values,
// which uu can't carry, since x * 50 / 50 isn't x for about one float in eight
int rlmax_get_bt(uint32_t nc, const uint32_t* state, uint32_t i, float* out) {
    if (i > nc) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        const auto* a = reinterpret_cast<const ArenaState<NC>*>(state);
        v3_to(i < nc ? a->cars[i].pos : a->ball.pos, out);
        v3_to(i < nc ? a->cars[i].vel : a->ball.vel, out + 3);
        return 0;
    });
}

// A car's boost alone: rlmax_set_car takes the body through uu and the basis, which moves it by a few ulp
int rlmax_set_boost(uint32_t nc, uint32_t* state, uint32_t ci, float boost) {
    if (ci >= nc) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        reinterpret_cast<ArenaState<NC>*>(state)->cars[ci].boost = boost;
        return 0;
    });
}

int rlmax_get_car(uint32_t nc, const uint32_t* state, uint32_t ci, RlmaxCar* out) {
    if (ci >= nc) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        const auto* a = reinterpret_cast<const ArenaState<NC>*>(state);
        const CarState& cs = a->cars[ci];
        body_out(cs.pos, cs.rot, cs.vel, cs.ang_vel, &out->body);
        if (seed_current(*a, ci + 1)) m3_to(a->seed_basis[ci + 1], out->body.rot);
        out->controls = controls_out(cs.ctl);
        out->flags = cs.flags;
        out->team = cs.team;
        out->preset = cs.body;
        out->boost = cs.boost;
        out->flip_torque[0] = cs.flip_torque_x;
        out->flip_torque[1] = cs.flip_torque_y;
        out->jump_ticks = cs.jump_ticks;
        out->flip_time = cs.flip_time;
        out->air_time = cs.air_time;
        out->time_since_jump = cs.time_since_jump;
        out->handbrake_val = cs.handbrake_val;
        out->supersonic_timer = cs.supersonic_timer;
        out->time_since_boosted = cs.time_since_boosted;
        out->boosting_time = cs.boosting_time;
        out->auto_flip_timer = cs.auto_flip_timer;
        out->auto_flip_torque_scale = cs.auto_flip_torque_scale;
        out->bump_cooldown = cs.bump_cooldown;
        out->bump_victim = cs.bump_victim;
        out->last_extra_hit_tick = cs.last_extra_hit_tick;
        out->demo_respawn_timer = cs.demo_respawn_timer;
        v3_to(cs.wheel_normal_sum, out->wheel_normal_sum);
        v3_to(cs.world_contact_normal, out->world_contact_normal);
        return 0;
    });
}

int rlmax_set_car(uint32_t nc, uint32_t* state, uint32_t ci, const RlmaxCar* in) {
    if (ci >= nc) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto* a = reinterpret_cast<ArenaState<NC>*>(state);
        CarState& cs = a->cars[ci];
        body_in(in->body, &cs.pos, &cs.rot, &cs.vel, &cs.ang_vel);
        seed_body(*a, ci + 1, in->body.rot, nullptr);
        cs.ctl = controls_in(in->controls);
        cs.flags = in->flags;
        cs.team = in->team;
        cs.body = in->preset < (uint32_t)NUM_BODY_PRESETS ? in->preset : 0u;
        cs.boost = in->boost;
        pad_frame_forget(cs, (int)ci);
        cs.flip_torque_x = in->flip_torque[0];
        cs.flip_torque_y = in->flip_torque[1];
        cs.jump_ticks = in->jump_ticks;
        cs.flip_time = in->flip_time;
        cs.air_time = in->air_time;
        cs.time_since_jump = in->time_since_jump;
        cs.handbrake_val = in->handbrake_val;
        cs.supersonic_timer = in->supersonic_timer;
        cs.time_since_boosted = in->time_since_boosted;
        cs.boosting_time = in->boosting_time;
        cs.auto_flip_timer = in->auto_flip_timer;
        cs.auto_flip_torque_scale = in->auto_flip_torque_scale;
        cs.bump_cooldown = in->bump_cooldown;
        cs.bump_victim = in->bump_victim;
        cs.last_extra_hit_tick = in->last_extra_hit_tick;
        cs.demo_respawn_timer = in->demo_respawn_timer;
        cs.wheel_normal_sum = v3_from(in->wheel_normal_sum);
        cs.world_contact_normal = v3_from(in->world_contact_normal);
        return 0;
    });
}

// Preserve a supplied basis and, when available, the inverse inertia tensor. Body indices are cars then ball.
int rlmax_set_body_basis(uint32_t nc, uint32_t* state, uint32_t body, const float* basis, const float* inverse) {
    if (!state || !basis || body > nc) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto& arena = *reinterpret_cast<ArenaState<NC>*>(state);
        const Q q = q_normalize(bullet_get_rotation(m3_from(basis)));
        if (body == nc) arena.ball.rot = q;
        else arena.cars[body].rot = q;
        seed_body(arena, body == nc ? 0u : body + 1u, basis, inverse);
        return 0;
    });
}

int rlmax_set_controls(uint32_t nc, uint32_t* state, uint32_t ci, const RlmaxControls* in) {
    if (ci >= nc) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        reinterpret_cast<ArenaState<NC>*>(state)->cars[ci].ctl = controls_in(*in);
        return 0;
    });
}

int rlmax_set_pad_frame_fraction(uint32_t nc, uint32_t* state, float fraction) {
    if (!state || !isfinite(fraction) || fraction > 1.0f) return -1;
    return with_nc(nc, [&](auto k) {
        reinterpret_cast<ArenaState<decltype(k)::value>*>(state)->pad_frame_bits = pad_encode_frac(fraction);
        return 0;
    });
}

uint32_t rlmax_tick_count(uint32_t nc, const uint32_t* state) {
    uint32_t t = 0;
    with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        t = reinterpret_cast<const ArenaState<NC>*>(state)->tick;
        return 0;
    });
    return t;
}

uint32_t rlmax_events(uint32_t nc, uint32_t* state, int clear) {
    uint32_t e = 0;
    with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto* a = reinterpret_cast<ArenaState<NC>*>(state);
        e = a->events;
        if (clear) a->events = 0;
        return 0;
    });
    return e;
}

uint32_t rlmax_contact_census(uint32_t nc, const uint32_t* state) {
    uint32_t bits = 0;
    with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        const auto* a = reinterpret_cast<const ArenaState<NC>*>(state);
        for (int i = 0; i < NC; i++)
            if (a->ball_man[i].n) bits |= 1u;
        for (int p = 0; p < NC * (NC - 1) / 2; p++)
            if (a->pair_man[p].n) bits |= 2u;
        return 0;
    });
    return bits;
}

int rlmax_local_sizes(uint32_t nc, uint32_t out[4]) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        constexpr int NB = 1 + NC, NP = NC * (NC - 1) / 2;
        out[0] = sizeof(EpaScratch);
        out[1] = sizeof(TickSolver<NC>);
        out[2] = sizeof(ContactPoint) *
                 (BALL_WORLD_SLOTS + NC * CAR_WORLD_SLOTS + NC * CAR_BALL_SLOTS + (NP > 0 ? NP : 1) * CAR_CAR_SLOTS);
        // per-body arrays, and the broadphase sweep's working set (its edge lists and the boxes it is given, sweep.h)
        out[3] = sizeof(Body) * NB + (sizeof(CarGeo) + sizeof(CarTickOut) + sizeof(CarSnap)) * NC + sizeof(Sweep<NC>) +
                 sizeof(SweepIn<NC>);
        return 0;
    });
}

static ArenaData arena_in(const RlmaxArenaData* arena) {
    ArenaData ad;
    ad.tris = reinterpret_cast<const Tri*>(arena->tris);
    ad.cell_off = arena->cell_off;
    ad.cell_idx = arena->cell_idx;
    ad.goal_ly = arena->goal_ly;
    ad.tri_box = arena->tri_box;
    ad.grid_min = v3(arena->grid_min[0], arena->grid_min[1], arena->grid_min[2]);
    ad.inv_cell = arena->inv_cell;
    ad.nx = arena->nx;
    ad.ny = arena->ny;
    ad.nz = arena->nz;
    ad.max_cell_count = arena->max_cell_count;
    return ad;
}

int rlmax_cpu_step(uint32_t nc, uint32_t* states, uint32_t n, uint32_t ticks, const RlmaxArenaData* arena) {
    ArenaData ad = arena_in(arena);
    return with_nc(nc, [&](auto k) { return dispatch_step<decltype(k)::value>(states, n, ticks, ad); });
}

int rlmax_refresh_sticky_gate_with_wheels(uint32_t nc, uint32_t* state, uint32_t ci, uint32_t wheels, const RlmaxArenaData* arena) {
    if (ci >= nc) return -1;
    ArenaData ad = arena_in(arena);
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto* a = reinterpret_cast<ArenaState<NC>*>(state);
        const TickConsts tk = tick_consts();
        Body bodies[1 + NC];
        CarGeo geo[NC];
        bodies[0] = make_body(a->ball.pos, a->ball.rot, a->ball.vel, a->ball.ang_vel, c::BALL_MASS_BT, tk.ball_inv_inertia);
        if (seed_current(*a, 0)) bodies[0].r = a->seed_basis[0];
        for (int i = 0; i < NC; i++) {
            const CarState& cs = a->cars[i];
            geo[i] = body_geo(cs.body);
            bodies[1 + i] = make_body(cs.pos, cs.rot, cs.vel, cs.ang_vel, c::CAR_MASS_BT, geo[i].inv_inertia_local);
            if (seed_current(*a, 1 + i)) bodies[1 + i].r = a->seed_basis[1 + i];
            bodies[1 + i].contact_response = !fl(cs, CF_IS_DEMOED);
        }
        CarState& cs = a->cars[ci];
        // The sticky normals are the previous tick's ray results, cast at the previous tick's pose. Position integrates
        // with the velocity after the update, so that pose is x - v dt, with the basis turned back by w dt.
        Body rb = bodies[1 + ci];
        rb.pos = rb.pos - rb.vel * c::DT;
        const float w_len = sqrtf(dot(rb.ang, rb.ang));
        if (w_len > 0.0f) {
            const V3 axis = rb.ang * (1.0f / w_len);
            const float back_sin = sinf(-w_len * c::DT), back_cos = cosf(-w_len * c::DT);
            const auto back = [&](V3 v) { return v * back_cos + cross(axis, v) * back_sin + axis * (dot(axis, v) * (1.0f - back_cos)); };
            rb.r.c0 = back(rb.r.c0);
            rb.r.c1 = back(rb.r.c1);
            rb.r.c2 = back(rb.r.c2);
        }
        int cell = cell_of(ad, rb.pos + rb.r * geo[ci].hitbox_off);
        bool any_world = false;
        uint32_t world_bits = 0;
        V3 nsum = v3s(0.0f);
        for (int w = 0; w < 4; w++) {
            if (!(wheels & (1u << w))) continue;
            V3 from = rb.pos + rb.r * geo[ci].wheel_conn[w];
            V3 to = from - rb.r.c2 * geo[ci].ray_len[w];
            WheelHit h = wheel_ray<NC>(ad, from, to, (int)ci, bodies, geo, tk.ball_radius, cell);
            if (h.ok && h.body < 0) {
                any_world = true;
                world_bits |= 1u << (20 + w);
                nsum += h.normal;
            }
        }
        set_fl(cs, CF_STICKY_GATE, any_world);
        cs.flags = (cs.flags & ~(0xFu << 20)) | world_bits;  // refresh_sticky_gate: world_wheels_prev
        cs.wheel_normal_sum = nsum;
        return 0;
    });
}

int rlmax_refresh_sticky_gate(uint32_t nc, uint32_t* state, uint32_t ci, const RlmaxArenaData* arena) {
    return rlmax_refresh_sticky_gate_with_wheels(nc, state, ci, 15, arena);
}

}  // extern "C"

// The world-only closest ray from `from` along `dir` for `len_bt`: the arena's planes and the triangles of the cell
// holding `from`, no minimum hit distance. The hit
// normal goes to `n`.
static bool world_ray(const ArenaData& ad, V3 from, V3 dir, float len_bt, V3* n) {
    const V3 to = from + dir * len_bt;
    RayHit best{1.0f, v3s(0), false};
    for (int p = 0; p < NUM_PLANES; p++) {
        V3 pn;
        V3 po;
        ray_plane_at(p, &po, &pn);
        ray_plane_game(from, to, po, pn, &best);
    }
    const int cell = cell_of(ad, from);
    for (uint32_t k = ad.cell_off[cell]; k < ad.cell_off[cell + 1]; k++) ray_tri(from, to, ad.tris[ad.cell_idx[k]], &best);
    if (best.ok) *n = best.n;
    return best.ok;
}

extern "C" {

// Seed car `ci`'s sticky gate and wheel normals from a recorded state's previous-step wheel contacts (`wheel_mask` bit w =
// wheel w touched, bit 5 they are the game's own world contacts, `normals` 4 x 3 their contact normals, both in RLMAX's wheel order). Keeps the wheels
// that touched the world and sets CF_STICKY_GATE, the world-wheel bits and wheel_normal_sum from them. Place every body
// first. Arithmetic in uu (BT times BT_TO_UU).
int rlmax_seed_wheel_contacts(uint32_t nc, uint32_t* state, uint32_t ci, uint32_t wheel_mask, const float* normals,
                              const RlmaxArenaData* arena) {
    if (ci >= nc) return -1;
    ArenaData ad = arena_in(arena);
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto* a = reinterpret_cast<ArenaState<NC>*>(state);
        const TickConsts tk = tick_consts();
        Body bodies[1 + NC];
        CarGeo geo[NC];
        bodies[0] = make_body(a->ball.pos, a->ball.rot, a->ball.vel, a->ball.ang_vel, c::BALL_MASS_BT, tk.ball_inv_inertia);
        if (seed_current(*a, 0)) bodies[0].r = a->seed_basis[0];
        for (int i = 0; i < NC; i++) {
            const CarState& cs = a->cars[i];
            geo[i] = body_geo(cs.body);
            bodies[1 + i] = make_body(cs.pos, cs.rot, cs.vel, cs.ang_vel, c::CAR_MASS_BT, geo[i].inv_inertia_local);
            if (seed_current(*a, 1 + i)) bodies[1 + i].r = a->seed_basis[1 + i];
            bodies[1 + i].contact_response = !fl(cs, CF_IS_DEMOED);
        }
        CarState& cs = a->cars[ci];
        const Body& rb = bodies[1 + ci];
        const float UU = c::BT_TO_UU;
        // the game's wheel ray is rest - 12 + 14.5 + radius, 2.5 uu past the ray RLMAX casts (ray_len = rest + radius)
        const float RG_EXTRA_UU = 14.5f - 12.0f;
        const int cell = cell_of(ad, rb.pos + rb.r * geo[ci].hitbox_off);
        WheelHit wh[4];
        V3 hp[4];
        for (int w = 0; w < 4; w++) {
            hp[w] = rb.pos + rb.r * geo[ci].wheel_conn[w];
            wh[w] = wheel_ray<NC>(ad, hp[w], hp[w] - rb.r.c2 * geo[ci].ray_len[w], (int)ci, bodies, geo, tk.ball_radius, cell);
        }
        float max_rest = 0.0f, max_rad = 0.0f;
        for (int w = 0; w < 4; w++) {
            max_rest = fmax_(max_rest, geo[ci].rest_len[w] + c::MAX_SUSPENSION_TRAVEL_UU * c::UU_TO_BT);
            max_rad = fmax_(max_rad, geo[ci].radius[w]);
        }
        const float other_max_len = 2.0f * (max_rest + max_rad) * UU;
        const V3 ball_uu = bodies[0].pos * UU;
        const float ball_radius_uu = tk.ball_radius * UU;
        uint32_t world_bits = 0;
        V3 nsum = v3s(0.0f);
        int n_kept = 0;
        const bool as_recorded = ((wheel_mask >> 5) & 1u) != 0;  // the game's own world contacts: nothing to test
        for (int w = 0; w < 4; w++) {
            if (!((wheel_mask >> w) & 1u)) continue;
            const V3 nrm = v3_from(normals + 3 * w);
            if (as_recorded) {
                world_bits |= 1u << (20 + w);
                nsum += nrm;
                n_kept++;
                continue;
            }
            const V3 hpu = hp[w] * UU;
            const float ray_uu = geo[ci].ray_len[w] * UU;
            const float reach = ray_uu + RG_EXTRA_UU + 30.0f;
            bool body_in_reach = len(hpu - ball_uu) < ball_radius_uu + reach;
            for (int j = 0; j < NC && !body_in_reach; j++) {
                if (j == (int)ci || !bodies[1 + j].contact_response) continue;
                const V3 center = (bodies[1 + j].pos + bodies[1 + j].r * geo[j].hitbox_off) * UU;
                body_in_reach = len(hpu - center) < len((geo[j].hitbox_half + v3s(geo[j].margin)) * UU) + reach;
            }
            if (body_in_reach) {
                // SetWheelNormalHasWorldFace: a ray from the hard point against the normal meets the arena on a face with it
                V3 hn = v3s(0.0f);
                const float flen_bt = (ray_uu + RG_EXTRA_UU + 20.0f) * c::UU_TO_BT;
                if (!(world_ray(ad, hp[w], -nrm, flen_bt, &hn) && dot(hn, nrm) > 0.9f * len(hn))) continue;
            }
            {
                const V3 to_wheel = hpu - ball_uu;
                const float dist = len(to_wheel);
                if (dist < ball_radius_uu + 80.0f && dist > 0.0f && dot(nrm, to_wheel) > 0.95f * dist) continue;
            }
            // the wheel's ray at the set pose lands on another car
            if (wh[w].ok && wh[w].body >= 1) continue;
            // SetWheelNormalIsOnOtherCar
            bool on_car = false;
            const V3 dir = -rb.r.c2;
            for (int j = 0; j < NC && !on_car; j++) {
                if (j == (int)ci || !bodies[1 + j].contact_response) continue;
                const Body& oj = bodies[1 + j];
                const V3 ax[3] = {oj.r.c0, oj.r.c1, oj.r.c2};
                const V3 half = (geo[j].hitbox_half + v3s(geo[j].margin)) * UU;
                const float grow = len((rb.vel - oj.vel) * UU) * (1.0f / 120.0f);
                const V3 center = (oj.pos + oj.r * geo[j].hitbox_off) * UU;
                const V3 d = hpu - center;
                float t_min = 0.0f, t_max = other_max_len;
                int face = -1;
                float face_sign = 0.0f;
                bool miss = false;
                for (int q = 0; q < 3 && !miss; q++) {
                    const float o = dot(d, ax[q]);
                    const float v = dot(dir, ax[q]);
                    const float hk = vget(half, q) + grow;
                    if (absf(v) < 1.0e-8f) {
                        miss = absf(o) > hk;
                        continue;
                    }
                    float t0 = (-hk - o) / v, t1 = (hk - o) / v;
                    float s0 = -1.0f;
                    if (t0 > t1) {
                        const float tmp = t0;
                        t0 = t1;
                        t1 = tmp;
                        s0 = 1.0f;
                    }
                    if (t0 > t_min) {
                        t_min = t0;
                        face = q;
                        face_sign = s0;
                    }
                    t_max = fmin_(t_max, t1);
                    miss = t_min > t_max;
                }
                if (miss || face < 0) continue;
                if (dot(ax[face] * face_sign, nrm) > 0.99f) {
                    on_car = true;
                    break;
                }
                // outward direction from the unexpanded box to the ray's entry point into the grown box
                const V3 p = d + dir * t_min;
                V3 outv = v3s(0.0f);
                for (int q = 0; q < 3; q++) {
                    const float o = dot(p, ax[q]);
                    const float hq = vget(half, q);
                    outv += ax[q] * (o - fmin_(fmax_(o, -hq), hq));
                }
                const float out_len = len(outv);
                if (out_len > 1.0e-4f && dot(outv, nrm) > 0.9f * out_len) on_car = true;
            }
            if (on_car) continue;
            world_bits |= 1u << (20 + w);
            nsum += nrm;
            n_kept++;
        }
        set_fl(cs, CF_STICKY_GATE, n_kept > 0);
        cs.flags = (cs.flags & ~(0xFu << 20)) | world_bits;
        cs.wheel_normal_sum = nsum;
        return 0;
    });
}

}  // extern "C"

#include "full_state.h"
#ifdef RLMAX_IMPULSE_LOG
// Impulse audit (lead 56, src/physics/impulse_log.h): car `ci`'s per-source velocity deltas of the last tick stepped.
// Layout of `out` (7 floats per entry: lin xyz, ang xyz, count): IL_N sources x {applied directly, accumulated}, then
// the wheel suspension entries (4 wheels) and the wheel friction entries (4 wheels); 2 * IL_N * 7 + 8 * 7 floats.
extern "C" void rlmax_seed_basis(uint32_t ci, const float* m) { g_seed_basis[ci] = M3{v3(m[0], m[1], m[2]), v3(m[3], m[4], m[5]), v3(m[6], m[7], m[8])}; g_seed_valid[ci] = true; }
extern "C" void rlmax_wheel_log(uint32_t ci, float* out) { memcpy(out, g_wheel_log[ci], sizeof(g_wheel_log[ci])); }
extern "C" int rlmax_impulse_log(uint32_t ci, float* out) {
    if (ci >= 8) return -1;
    const ImpLog& g = g_imp_log[ci];
    auto put = [&](const ImpEntry& e) {
        for (int k = 0; k < 3; k++) *out++ = e.lin[k];
        for (int k = 0; k < 3; k++) *out++ = e.ang[k];
        *out++ = e.n;
    };
    for (int s = 0; s < IL_N; s++)
        for (int a = 0; a < 2; a++) put(g.e[s][a]);
    for (int k = 0; k < 2; k++)
        for (int w = 0; w < 4; w++) put(g.w[k][w]);
    return 0;
}
#endif

// Test hooks for the deterministic transcendental functions (crates/rlmax/src/cpu.rs tests).
extern "C" void rlmax_test_sincos(float x, float* s, float* c) { sincos_poly(x, s, c); }
extern "C" float rlmax_test_atan2(float y, float x) { return atan2_(y, x); }

// Debug: the car-car contacts RLMAX would generate for cars 0 and 1 at the current pose (no stepping).
extern "C" int rlmax_debug_car_car(uint32_t nc, const uint32_t* state, float* out /* 4 x (pb3, n3, d, ok) */) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        const auto* a = reinterpret_cast<const ArenaState<NC>*>(state);
        const CarGeo g0 = body_geo(a->cars[0].body), g1 = body_geo(a->cars[1].body);
        Body b0 = make_body(a->cars[0].pos, a->cars[0].rot, a->cars[0].vel, a->cars[0].ang_vel, c::CAR_MASS_BT, g0.inv_inertia_local);
        Body b1 = make_body(a->cars[1].pos, a->cars[1].rot, a->cars[1].vel, a->cars[1].ang_vel, c::CAR_MASS_BT, g1.inv_inertia_local);
        Contact cc[4];
        obb_obb<4>(car_obb(b0, g0), car_obb(b1, g1), fmin_(g0.contact_threshold, g1.contact_threshold), cc);
        for (int i = 0; i < 4; i++) {
            float* o = out + i * 8;
            o[0] = cc[i].pb.x * c::BT_TO_UU; o[1] = cc[i].pb.y * c::BT_TO_UU; o[2] = cc[i].pb.z * c::BT_TO_UU;
            o[3] = cc[i].n.x; o[4] = cc[i].n.y; o[5] = cc[i].n.z;
            o[6] = cc[i].d * c::BT_TO_UU; o[7] = cc[i].ok ? 1.0f : 0.0f;
        }
        return 0;
    });
}

// Debug: RLMAX's car-world contacts for car `ci` at the current pose (4 slots, same layout as rlmax_debug_car_car).
extern "C" int rlmax_debug_car_world(uint32_t nc, const uint32_t* state, uint32_t ci, const RlmaxArenaData* arena, float* out) {
    ArenaData ad = arena_in(arena);
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        const auto* a = reinterpret_cast<const ArenaState<NC>*>(state);
        const CarState& cs = a->cars[ci];
        Body b = make_body(cs.pos, cs.rot, cs.vel, cs.ang_vel, c::CAR_MASS_BT, body_geo(cs.body).inv_inertia_local);
        const CarGeo g = body_geo(cs.body);
        Obb ob = car_obb(b, g);
        Contact cw[4];
        for (int j = 0; j < 4; j++) cw[j] = no_contact();
        for (int p = 0; p < NUM_PLANES; p++) {
            V3 pn; float pd;
            arena_plane(p, &pn, &pd);
            Contact four[4];
            obb_plane4(ob, pn, pd, g.contact_threshold, four);
            for (int q = 0; q < 4; q++) keep_deepest<4>(cw, four[q]);
        }
        int cell = cell_of(ad, ob.c);
        for (uint32_t t = ad.cell_off[cell]; t < ad.cell_off[cell + 1]; t++) {
            const Tri& tri = ad.tris[ad.cell_idx[t]];
            Contact cc = obb_tri(ob, tri, ad.goal_ly, g.contact_threshold, cpu_scratch());
            if (cc.ok) adjust_internal_edge(tri, &cc, ad.goal_ly);
            keep_deepest<4>(cw, cc);
        }
        for (int i = 0; i < 4; i++) {
            float* o = out + i * 8;
            o[0] = cw[i].pb.x * c::BT_TO_UU; o[1] = cw[i].pb.y * c::BT_TO_UU; o[2] = cw[i].pb.z * c::BT_TO_UU;
            o[3] = cw[i].n.x; o[4] = cw[i].n.y; o[5] = cw[i].n.z;
            o[6] = cw[i].d * c::BT_TO_UU; o[7] = cw[i].ok ? 1.0f : 0.0f;
        }
        return 0;
    });
}

// Debug: RLMAX's ball-world contacts at the current pose, as the tick builds them. `raw` gets every contact found, in
// the order found (up to raw_cap; *raw_n is the total), `manifold` the ball's world manifolds as the last tick left them
// (BALL_WORLD_MANIFOLDS x 4 slots). Layout per
// contact: point on the world (uu), normal, depth (uu), group (mesh index, or 0x100 + plane), triangle index (planes ~0).
extern "C" int rlmax_debug_ball_world(uint32_t nc, const uint32_t* state, const RlmaxArenaData* arena, float* raw,
                                      uint32_t raw_cap, uint32_t* raw_n, float* manifold, uint32_t* cell_count) {
    ArenaData ad = arena_in(arena);
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        const auto* a = reinterpret_cast<const ArenaState<NC>*>(state);
        const TickConsts tk = tick_consts();
        V3 bc = a->ball.pos;
        auto put = [](float* o, const Contact& cc, uint32_t tri) {
            o[0] = cc.pb.x * c::BT_TO_UU; o[1] = cc.pb.y * c::BT_TO_UU; o[2] = cc.pb.z * c::BT_TO_UU;
            o[3] = cc.n.x; o[4] = cc.n.y; o[5] = cc.n.z;
            o[6] = cc.d * c::BT_TO_UU; o[7] = (float)cc.group; o[8] = (float)tri;
        };
        uint32_t n = 0;
        for (int p = 0; p < NUM_PLANES; p++) {
            Contact cp = sphere_plane(bc, bullet_set_rotation(a->ball.rot), tk.ball_radius, plane_xf(p), tk.ball_threshold);
            cp.group = 0x100u + (uint32_t)p;
            if (cp.ok && n < raw_cap) put(raw + n * 9, cp, ~0u);
            if (cp.ok) n++;
        }
        int cell = cell_of(ad, bc);
        *cell_count = ad.cell_off[cell + 1] - ad.cell_off[cell];
        for (uint32_t t = ad.cell_off[cell]; t < ad.cell_off[cell + 1]; t++) {
            const Tri& tri = ad.tris[ad.cell_idx[t]];
            Contact cc = sphere_tri(tri, bc, tk.ball_radius, tk.ball_threshold, ad.goal_ly);
            cc.group = tri_mesh(tri);
            if (cc.ok) adjust_internal_edge(tri, &cc, ad.goal_ly);
            if (cc.ok && n < raw_cap) put(raw + n * 9, cc, ad.cell_idx[t]);
            if (cc.ok) n++;
        }
        *raw_n = n;
        const auto& w = a->ball_wman;
        for (int i = 0; i < BALL_WORLD_SLOTS; i++) {
            const int q = i / MANIFOLD_CAPACITY, j = i % MANIFOLD_CAPACITY;
            const bool ok = q < (int)w.n && j < (int)w.m[q].n;
            float* o = manifold + i * 9;
            for (int e = 0; e < 9; e++) o[e] = 0.0f;
            if (ok) {
                const V3 org = mesh_origin(w.obj[q]);
                Contact cc = man_point(w.m[q], nullptr, j, nullptr, w.obj[q] - 4u < 4u ? &org : nullptr);
                cc.group = w.obj[q];
                put(o, cc, 0);
            }
            o[8] = ok ? 1.0f : 0.0f;
        }
        return 0;
    });
}

// Debug: RLMAX's raw car-ball contact for one car at the current pose: point on the car (uu), normal, depth (uu), ok.
extern "C" int rlmax_debug_car_ball(uint32_t nc, const uint32_t* state, uint32_t car, float* out) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        const auto* a = reinterpret_cast<const ArenaState<NC>*>(state);
        if (car >= (uint32_t)NC) return 1;
        const TickConsts tk = tick_consts();
        const CarState& cs = a->cars[car];
        const CarGeo g = body_geo(cs.body);
        Body b = make_body(cs.pos, cs.rot, cs.vel, cs.ang_vel, c::CAR_MASS_BT, g.inv_inertia_local);
        Contact cc = sphere_obb(a->ball.pos, tk.ball_radius, car_obb(b, g), fmin_(tk.ball_threshold, g.contact_threshold));
        out[0] = cc.pb.x * c::BT_TO_UU; out[1] = cc.pb.y * c::BT_TO_UU; out[2] = cc.pb.z * c::BT_TO_UU;
        out[3] = cc.n.x; out[4] = cc.n.y; out[5] = cc.n.z;
        out[6] = cc.d * c::BT_TO_UU; out[7] = cc.ok ? 1.0f : 0.0f;
        return 0;
    });
}

// Debug: one car's world manifolds as the last tick left them, CAR_WORLD_MANIFOLDS x 4 slots. Layout per slot: point on
// the world (uu), normal, distance (uu), static object (mesh index, or 0x100 + plane), ok.
extern "C" int rlmax_debug_car_wman(uint32_t nc, const uint32_t* state, uint32_t car, float* out) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        const auto* a = reinterpret_cast<const ArenaState<NC>*>(state);
        if (car >= (uint32_t)NC) return 1;
        const auto& w = a->cars[car].wman;
        for (int i = 0; i < CAR_WORLD_MANIFOLDS * MANIFOLD_CAPACITY; i++) {
            const int q = i / MANIFOLD_CAPACITY, j = i % MANIFOLD_CAPACITY;
            const bool ok = q < (int)w.n && j < (int)w.m[q].n;
            float* o = out + i * 9;
            for (int e = 0; e < 9; e++) o[e] = 0.0f;
            if (!ok) continue;
            const MPoint& p = w.m[q].p[j];
            o[0] = p.lb.x * c::BT_TO_UU; o[1] = p.lb.y * c::BT_TO_UU; o[2] = p.lb.z * c::BT_TO_UU;
            o[3] = p.n.x; o[4] = p.n.y; o[5] = p.n.z;
            o[6] = p.d * c::BT_TO_UU; o[7] = (float)w.obj[q]; o[8] = 1.0f;
        }
        return 0;
    });
}

// Debug: the car pair (0, 1) manifold as the last tick left it, 4 slots: point on car 1 (world, uu), normal on car 1,
// distance (uu), ok.
extern "C" int rlmax_debug_pair_man(uint32_t nc, const uint32_t* state, float* out) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        const auto* a = reinterpret_cast<const ArenaState<NC>*>(state);
        const Manifold& m = a->pair_man[0];
        const CarState& cb = a->cars[1];
        Body b = make_body(cb.pos, cb.rot, cb.vel, cb.ang_vel, c::CAR_MASS_BT, body_geo(cb.body).inv_inertia_local);
        for (int i = 0; i < MANIFOLD_CAPACITY; i++) {
            float* o = out + i * 8;
            for (int e = 0; e < 8; e++) o[e] = 0.0f;
            if (i >= (int)m.n) continue;
            Contact cc = man_point(m, &b, i);
            o[0] = cc.pb.x * c::BT_TO_UU; o[1] = cc.pb.y * c::BT_TO_UU; o[2] = cc.pb.z * c::BT_TO_UU;
            o[3] = cc.n.x; o[4] = cc.n.y; o[5] = cc.n.z;
            o[6] = m.p[i].d * c::BT_TO_UU; o[7] = 1.0f;
        }
        return 0;
    });
}
