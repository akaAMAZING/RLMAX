    RLMAX_PROF(5)
    // a wheel on the ball activates it, as the ball's applyImpulse does: a sleeping one is solved this tick with its
    // world contacts, without this tick's gravity, and either way its deactivation timer restarts
#ifdef RLMAX_STAGED_REFERENCES
    bool& wheels_woke_ball = work.wheels_woke_ball;
    wheels_woke_ball = false;
    V3 (&predict_lin)[NB] = work.predict_lin;
    V3 (&predict_ang)[NB] = work.predict_ang;
#else
    bool wheels_woke_ball = false;
    V3 predict_lin[NB], predict_ang[NB];  // the velocities the broadphase predicts with: after the pre-ticks, before the wheels
#endif
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
            vehicle_update<NC>(i, cs, ctx, cs.handbrake_val, outs[i].real_throttle, outs[i]);
            wheels_woke_ball = wheels_woke_ball || (outs[i].pushed & PUSHED_BALL) != 0;
            // the wheels' ground-stick pushed a car that has yet to take its turn: its drive follows the new velocity
            if (outs[i].pushed != 0)
                for (int j = 0; j < NC; j++)
                    if ((outs[i].pushed >> j & 1u) && !(visited >> j & 1u) && pre_ticked[j]) car_redrive<NC, false>(j, s.cars[j], ctx, outs[j]);
        }
    }

    if (wheels_woke_ball) {
        bs.deactivation_time = 0.0f;
        if (bs.sleeping == BALL_WANTS_DEACTIVATION) bs.sleeping = BALL_ACTIVE;
    }
