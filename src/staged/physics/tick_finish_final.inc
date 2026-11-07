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
    if (s.countdown) countdown_advance(s.countdown, s.countdown_count, s.pad_frame_bits);

    const float goal_y = (c::GOAL_THRESHOLD_Y_UU + c::BALL_RADIUS_UU) * c::UU_TO_BT;
    if (bs.pos.y > goal_y) s.events |= 1u;   // in orange's goal: blue scored
    if (bs.pos.y < -goal_y) s.events |= 2u;  // in blue's goal: orange scored
    RLMAX_PROF(17)