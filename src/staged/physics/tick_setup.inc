    RLMAX_PROF_BEGIN
#ifdef __CUDA_ARCH__
    // the sweep reads its state late in the tick, from an arena 3.5 KB from its warp neighbours' (one line each, DRAM
    // latency): ask L2 for its lines now, so the reads there hit (P3 §6, the sweep's sub-profile)
    for (int o = 0; o < (int)sizeof(s.sweep); o += 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<const char*>(&s.sweep) + o));
#endif
    if (s.countdown) countdown_hold<NC>(s);  // the round countdown: every car undriven
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
#ifdef RLMAX_STAGED_REFERENCES
    Body (&bodies)[NB] = work.bodies;
#else
    Body bodies[NB];
#endif

    RLMAX_PROF(0)
    // ---- bodies from state
    BallState& bs = s.ball;
    const bool ball_uploaded = bs.sleeping == BALL_UPLOADED;
    if (ball_uploaded) resolve_uploaded_ball(bs);
    bodies[0] = make_body(bs.pos, bs.rot, bs.vel, bs.ang_vel, c::BALL_MASS_BT, k.ball_inv_inertia);
    if (ball_uploaded && bs.sleeping == BALL_SLEEPING) prime_uploaded_ball_planes(s.ball_wman, bodies[0], k.ball_radius, k.ball_threshold);
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
    if (s.seed_mask) {
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
#ifdef RLMAX_STAGED_REFERENCES
    V3 (&pre_q_pos)[NB] = work.pre_q_pos;
#else
    V3 pre_q_pos[NB];
#endif
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
#ifdef RLMAX_STAGED_REFERENCES
    work.ball_snap_vel_uu = bodies[0].vel_uu;
#else
    CarSnap snap[NC];
    const V3 ball_snap_vel_uu = bodies[0].vel_uu;
#endif

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
