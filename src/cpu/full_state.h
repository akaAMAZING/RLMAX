// Test-only full-state loader FFI (cargo feature `full_state`; included at the end of cpu.cpp). Loads what the suite records
// beyond the public state, straight into the arena in Bullet units, so the first tick starts from the game's own floats.
#pragma once
#ifdef RLMAX_FULL_STATE

namespace {
inline V3 fs_v3(const float* p) { return v3(p[0], p[1], p[2]); }
}  // namespace

extern "C" {
// Every pad given this tick, so none can give on the next: the suite's boost comes from the game's recorded pickups, and the card build
// has no test define to turn its pads off
int rlmax_fs_block_pads(uint32_t nc, uint32_t* state) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto* a = reinterpret_cast<ArenaState<NC>*>(state);
        for (int p = 0; p < NUM_PADS; p++) a->pad_gave_tick[p] = a->tick;
        return 0;
    });
}
// Controls-only runs start from the game's pad state. Pad `p` (RLMAX's pad_def order): the game's remaining cooldown in seconds, whether it
// is active, and its quantised pickup fraction. The game's cooldown is decremented before each tick's respawn test, which fires when the
// remaining ticks, rounded, are below zero, or zero with the pickup fraction under the frame's; pads_update's age test is the same with
// age = period - remaining at the tick that starts from here, so the pad gave `period - remaining` ticks before the arena's tick. Call after
// rlmax_fs_car_extra (which sets that tick). The game's cur/prev locked car ids have no home: pads_update locks within a tick, and the game
// clears cur_locked at the start of the next one, so neither is read across the boundary.
int rlmax_fs_pad(uint32_t nc, uint32_t* state, uint32_t p, float cooldown, float active, float frac_q) {
    if (p >= (uint32_t)NUM_PADS) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto* a = reinterpret_cast<ArenaState<NC>*>(state);
        a->pad_pickup_frac[p] = (uint16_t)frac_q;
        if (active != 0.0f && !(cooldown > 0.0f)) { a->pad_gave_tick[p] = PAD_NONE; return 0; }
        const uint32_t period = pad_def((int)p).big ? PAD_COOLDOWN_BIG_TICKS : PAD_COOLDOWN_SMALL_TICKS;
        // the recordings mark a pad that never respawns with a cooldown of 1e9 s; the arena counts at most one period, which outlasts a segment
        const uint32_t remaining = (uint32_t)(int32_t)fminf(floorf(cooldown / c::DT + 0.5f), (float)period);
        const uint32_t gave = a->tick - period + remaining;
        if (gave == PAD_NONE) return -3;  // the "never" word: this pad would read as active
        a->pad_gave_tick[p] = gave;
        return 0;
    });
}
// The game's round countdown at the start state: phase (0 none, 1 first second, 2 the three fires), fires so far, and its float counter
// (EnvState.countdown_phase, countdown_fires, countdown_count). The recorded frame fraction (rlmax_set_pad_frame_fraction) is what each
// tick adds to the counter. Phase 0 clears it.
int rlmax_fs_countdown(uint32_t nc, uint32_t* state, uint32_t phase, uint32_t fires, float count) {
    if (phase > 2 || fires > 2 || !isfinite(count)) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto* a = reinterpret_cast<ArenaState<NC>*>(state);
        a->countdown = phase ? countdown_word(phase, fires) : 0u;
        a->countdown_count = phase ? count : 0.0f;
        return 0;
    });
}
// car `ci`'s pad state: the pad it touches (RLMAX's index, or -1 none), and the last render-frame end position when the game holds one.
// Clears the car's cached pad cell, and slot 0's expiry deadline, which the next step recomputes.
int rlmax_fs_car_pad(uint32_t nc, uint32_t* state, uint32_t ci, int32_t touching, float frame_valid, const float* frame_pos_bt) {
    if (ci >= nc) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        CarState& cs = reinterpret_cast<ArenaState<NC>*>(state)->cars[ci];
        cs.pad_touch = touching < 0 ? PAD_NONE : (uint32_t)touching;
        cs.pad_frame_valid = frame_valid != 0.0f ? 1u : 0u;
        cs.pad_frame_pos = fs_v3(frame_pos_bt);
        return 0;
    });
}
// car `ci`'s pad_touch (RLMAX's pad index, 0xFFFFFFFF none), for diagnosis
uint32_t rlmax_fs_pad_touch(uint32_t nc, const uint32_t* state, uint32_t ci) {
    uint32_t t = PAD_NONE;
    with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        if (ci < (uint32_t)NC) t = reinterpret_cast<const ArenaState<NC>*>(state)->cars[ci].pad_touch;
        return 0;
    });
    return t;
}
// the pads holding a pickup (gave a boost and not yet respawned), bit p = RLMAX's pad p
uint64_t rlmax_fs_pads_taken(uint32_t nc, const uint32_t* state) {
    uint64_t m = 0;
    with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        const auto* a = reinterpret_cast<const ArenaState<NC>*>(state);
        for (int p = 0; p < NUM_PADS; p++) m |= (uint64_t)(a->pad_gave_tick[p] != PAD_NONE) << p;
        return 0;
    });
    return m;
}
// the game's own jump state: jumped, and the time since the car last jumped
int rlmax_fs_jump(uint32_t nc, uint32_t* state, uint32_t ci, float jumped, float used, float inactive, float ground) {
    if (ci >= nc) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        CarState& cs = reinterpret_cast<ArenaState<NC>*>(state)->cars[ci];
        set_fl(cs, CF_HAS_JUMPED, jumped != 0.0f);
        set_fl(cs, CF_AIR_JUMP_USED, used != 0.0f);
        cs.time_since_jump = inactive;
        set_fl(cs, CF_JUMP_GROUND, ground != 0.0f);
        return 0;
    });
}
// the arena runs its own boost pads (controls-only recordings; the default of the test build is none, the credits come from the recording)
void rlmax_fs_pads_on(int on) { g_fs_pads = on != 0; }
void rlmax_fs_spawn_z(uint32_t ci, float z) { if (ci < 8) g_fs_spawn_z[ci] = z; }
int rlmax_seed_handle(uint32_t nc, uint32_t* state, uint32_t body, uint32_t handle) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        reinterpret_cast<ArenaState<NC>*>(state)->sweep.handle[body] = (uint8_t)handle;
        return 0;
    });
}
// car `ci`: float block of 28 (see fs_seed)
int rlmax_fs_car(uint32_t nc, uint32_t* state, uint32_t ci, const float* f) {
    if (ci >= nc) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto& arena = *reinterpret_cast<ArenaState<NC>*>(state);
        CarState& cs = arena.cars[ci];
        cs.pos = fs_v3(f); cs.vel = fs_v3(f + 3); cs.ang_vel = fs_v3(f + 6);
        seed_body(arena, ci + 1, f + 9, f[27] != 0.0f ? f + 18 : nullptr);
        cs.wman.n = 0;
        return 0;
    });
}
// car `ci`'s hidden scalars the public state lacks, from the recorded state: world contact present, its normal, the last tick's jump input, the bump cooldown and its victim, the recorded respawn spot
int rlmax_fs_car_extra(uint32_t nc, uint32_t* state, uint32_t ci, const float* e) {
    if (ci >= nc) return -1;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        CarState& cs = reinterpret_cast<ArenaState<NC>*>(state)->cars[ci];
        set_fl(cs, CF_WORLD_CONTACT, e[0] != 0.0f);
        set_fl(cs, CF_PREV_JUMP, e[4] != 0.0f);
        // the game's car-contact cooldown (the public row carries none), and who it blocks
        if (e[5] > 0.0f) {
            cs.bump_cooldown = e[5];
            set_fl(cs, CF_HAS_BUMP_VICTIM, true);
            cs.bump_victim = (uint32_t)e[6];
        }
        g_fs_respawn[ci] = (uint8_t)e[7];
        // the arena's tick counter: the extra-hit rule compares the recorded absolute ticks of a car's last hit with it
        reinterpret_cast<ArenaState<NC>*>(state)->tick = (uint32_t)e[8];
        // the last ball hit's extra-impulse tick, valid with a prior hit
        set_fl(cs, CF_HAS_EXTRA_HIT, e[9] != 0.0f);
        cs.last_extra_hit_tick = (uint32_t)e[10];
        cs.world_contact_normal = fs_v3(e + 1);
        return 0;
    });
}
// the ball: float block of 28, then sleeping state and deactivation time
int rlmax_fs_ball(uint32_t nc, uint32_t* state, const float* f) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto& arena = *reinterpret_cast<ArenaState<NC>*>(state);
        BallState& bs = arena.ball;
        bs.pos = fs_v3(f); bs.vel = fs_v3(f + 3); bs.ang_vel = fs_v3(f + 6);
        // the recording's numbering: 0 awake, 1 asleep, 2 uploaded; awake past the timer asks to sleep
        const uint32_t rec = (uint32_t)f[28] & 3u;
        bs.deactivation_time = f[29];
        bs.sleeping = rec == 1u ? BALL_SLEEPING : rec == 2u ? BALL_UPLOADED : f[29] > 2.0f ? BALL_WANTS_DEACTIVATION : BALL_ACTIVE;
        seed_body(arena, 0, f + 9, f[27] != 0.0f ? f + 18 : nullptr);
        bs.sleep_inv_i = f[27] != 0.0f ? arena.seed_inv[0] : world_inv_inertia(arena.seed_basis[0], tick_consts().ball_inv_inertia);
        reinterpret_cast<ArenaState<NC>*>(state)->ball_wman.n = 0;
        return 0;
    });
}
// A recorded manifold (stage 3): kind 0 a car's world manifold (a = car, obj = mesh index or 0x100 + plane; the normal of a plane point is set here), 1 the ball's world
// manifold, 2 a car pair (a = pair index, A = the lower car), 3 the ball against car a. `pts`: count points of
// (point on A in A's frame, point on B in B's frame, normal on B, distance), Bullet units. World manifolds are appended in call order.
int rlmax_fs_manifold(uint32_t nc, uint32_t* state, uint32_t kind, uint32_t a, uint32_t obj, uint32_t count, const float* pts) {
    if (count > (uint32_t)MANIFOLD_CAPACITY) return -2;
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        auto* ar = reinterpret_cast<ArenaState<NC>*>(state);
        Manifold* m = nullptr;
        if (kind == 0 && a < (uint32_t)NC) m = wman_get(ar->cars[a].wman, obj);
        else if (kind == 1) m = wman_get(ar->ball_wman, obj);
        else if (kind == 2 && a < (uint32_t)(NC * (NC - 1) / 2)) m = &ar->pair_man[a];
        else if (kind == 3 && a < (uint32_t)NC) m = &ar->ball_man[a];
        if (!m) return -1;
        m->n = count;
        // a car's plane point is recorded without its normal: the plane's own, which plane_point sets
        const PlaneXf px = plane_xf(kind == 0 && obj >= 0x100u ? (int)(obj - 0x100u) : 0);
        for (uint32_t i = 0; i < count; i++) {
            const float* q = pts + 10 * i;
            // the ball's plane points are recorded as world points; RLMAX keeps them from the plane's origin
            const V3 lb = kind == 1 && obj >= 0x100u ? fs_v3(q + 3) - plane_xf((int)(obj - 0x100u)).o : fs_v3(q + 3);
            // a recorded car-car normal runs from A to B (point on B = point on A + normal * distance); RLMAX keeps the normal on B
            const V3 nr = kind == 2 ? -fs_v3(q + 6) : fs_v3(q + 6);
            m->p[i] = MPoint{fs_v3(q), lb, kind == 0 && obj >= 0x100u ? v3(px.r0.z, px.r1.z, px.r2.z) : nr, q[9]};
        }
        return 0;
    });
}
// The recorded visit order: the game's packed order (a nibble per slot, 8 slots) and its present-slot mask. The slots past the
// car count are dropped and absent cars (demolished) sit last, as the game keeps them.
int rlmax_fs_visit(uint32_t nc, uint32_t* state, uint32_t packed, uint32_t present) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        uint32_t v = 0, off = 0;
        int n = 0;
        for (int q = 0; q < 8; q++) {
            const uint32_t slot = packed >> (4 * q) & 15u;
            if (slot >= (uint32_t)NC) continue;
            v |= (slot ^ (uint32_t)n) << (visit_bits<NC>() * n++);
            off |= (present >> slot & 1u ? 0u : 1u) << slot;
        }
        reinterpret_cast<ArenaState<NC>*>(state)->visit = v | off << 24;
        return n == NC ? 0 : -1;
    });
}
// The free handles alone, for a recorded state whose broadphase can't be loaded (a demolished car left the sweep): popped last first.
int rlmax_fs_free(uint32_t nc, uint32_t* state, uint32_t n, const float* f) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        if (n > (uint32_t)(1 + NC)) return -2;
        SweepState<NC>& sw = reinterpret_cast<ArenaState<NC>*>(state)->sweep;
        sw.free_n = (uint8_t)n;
        for (uint32_t i = 0; i < n; i++) sw.free_handles[i] = (uint8_t)f[i];
        return 0;
    });
}
// The recorded broadphase: np, nm, free_n, the free handles
// (1 + nc), per body (ball, then cars) six list places (axis * 2 + max) and six positions (the same order), then the np pair
// keys and the nm manifold keys in order (sw_key). Marks the sweep valid, so the first tick continues from it instead of rebuilding.
int rlmax_fs_sweep(uint32_t nc, uint32_t* state, const float* f) {
    return with_nc(nc, [&](auto k) {
        constexpr int NC = decltype(k)::value;
        SweepState<NC>& sw = reinterpret_cast<ArenaState<NC>*>(state)->sweep;
        const uint32_t np_raw = (uint32_t)f[0], np = np_raw & 255u, nm = (uint32_t)f[1];
        if (np > (uint32_t)sweep_cap<NC>() || nm > (uint32_t)sweep_man_cap<NC>()) return -2;
        sw.free_n = (uint8_t)f[2];
        const float* q = f + 3;
        for (int i = 0; i < 1 + NC; i++) sw.free_handles[i] = (uint8_t)*q++;
        for (int b = 0; b < 1 + NC; b++) {
            for (int j = 0; j < 6; j++) sw.slot[b][j] = (uint8_t)*q++;
            for (int a = 0; a < 3; a++, q += 2) sw.edge[b][a] = (uint32_t)q[0] | ((uint32_t)q[1] << 16);
        }
        for (uint32_t i = 0; i < np; i++) sw.pairs[i] = (SweepKey<NC>)*q++;
        for (uint32_t i = 0; i < nm; i++) sw.man[i] = (SweepKey<NC>)*q++;
        sw.hdr = (np_raw >= 256u ? 8u : 1u) | (np << 8) | (nm << 16);
        return 0;
    });
}
}
#endif
