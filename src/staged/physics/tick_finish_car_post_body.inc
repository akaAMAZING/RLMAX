    CarState& cs = s.cars[i];
    Body& rb = bodies[1 + i];
    if (fl(cs, CF_IS_DEMOED)) {
        // the game keeps the body where it was; the car is hidden until it respawns
        cs.pos = rb.pos;
        cs.rot = rb.rot;
    } else {
        car_post_tick(cs, rb, outs[i]);
        rb.vel = bumped_vel[i];
        cs.pos = rb.pos;
        cs.rot = rb.rot;
        cs.vel = rb.vel;
        cs.ang_vel = rb.ang;
    }
