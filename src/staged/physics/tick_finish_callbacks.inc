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
    V3* bumped_vel = work.post_bumped_vel;
    // Apply pending bumps to velocity in hitter order.
    for (int i = 0; i < NC; i++) bumped_vel[i] = bodies[1 + i].vel;
    for (int i = 0; i < NC; i++) {
        if (!car_ball_touch[i]) continue;
        CarState& cs = s.cars[i];
        // Car_TA.ApplyBallImpactForces takes ForwardDir = GetRotatorAxis(QuatToRotator(OldRBState.Quaternion), 0): yaw
        // and pitch in integer UE3 units (appRound: int(rint(2v + 0.5)) >> 1), the axis rebuilt from UE3's 16384-entry
        // sine table (index angle >> 2)
        // the rotator from the tick-start quaternion by the game's atan2f, the uu state, Ue3SafeNormal at each step
        const Q q_ = snap[i].q;
        const float x2_ = q_.x + q_.x, y2_ = q_.y + q_.y, z2_ = q_.z + q_.z;
        const V3 f = v3(1.0f - (q_.y * y2_ + q_.z * z2_), q_.x * y2_ + q_.w * z2_, q_.x * z2_ - q_.w * y2_);
        (void)x2_;
        auto ue3_round = [](float v) { return ((int)rintf(v + v + 0.5f)) >> 1; };
        const int fwd_yaw = ue3_round(game_atan2f(f.y, f.x) * 32768.0f / 3.14159274f);
        const float fxy_ = sqrtf(f.y * f.y + f.x * f.x);
        const int fwd_pitch = ue3_round(game_atan2f(f.z, fxy_) * 32768.0f / 3.14159274f);
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
            bodies[0].vel += ((imp * 0.02f) * bodies[0].mass) * bodies[0].inv_mass;
            cs.last_extra_hit_tick = s.tick;
            set_fl(cs, CF_HAS_EXTRA_HIT, true);
        }
    }
    if (any_car_car) staged_car_car_callbacks<NC>(s, snap, geo, car_car, bumped_vel, scratch);
    for (int i = 0; i < NC; i++) {
        if (!car_world_hit[i]) continue;
        s.cars[i].world_contact_normal = car_world_normal[i];
        set_fl(s.cars[i], CF_WORLD_CONTACT, true);
    }
