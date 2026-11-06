#pragma once
#include "../staged_tick.h"
namespace rlmax {
// Gravity and the independence gate. Dependent arenas leave vehicle
// work for the next stage, which calls the ordered fallback.
template<int NC>
RLMAX_HD void split_vehicle_gate(ArenaState<NC>& s, StagedTick<NC>& work) {
    if (s.ball.sleeping != BALL_SLEEPING) {
        Body& ball = work.bodies[0];
        ball.acc_lin += tick_consts().gravity * c::DT;
        ball.vel = ball.vel * BALL_DAMPING_PER_TICK;
    }
    work.independent = quirk::VEHICLE_IN_STEP && (s.sweep.hdr & 2u);
    for (int i = 0; i < NC; ++i) work.independent = work.independent && work.car_independent[i];
    if (!work.independent) return;
    for (int i = 0; i < NC; ++i)
        if (work.bodies[1 + i].active)
            limit_vels(work.bodies[1 + i], c::CAR_MAX_SPEED_UU * c::UU_TO_BT, c::CAR_MAX_ANG_SPEED);
    if (s.ball.sleeping != BALL_SLEEPING)
        limit_vels(work.bodies[0], c::BALL_MAX_SPEED_UU * c::UU_TO_BT, c::BALL_MAX_ANG_SPEED);
    for (int i = 0; i < NC + 1; ++i) {
        work.predict_lin[i] = work.bodies[i].vel;
        work.predict_ang[i] = work.bodies[i].ang;
    }
    for (int i = 0; i < NC; ++i) if (work.pre_ticked[i]) work.outs[i].pushed = 0;
    work.wheels_woke_ball = false;
}
template<int NC>
RLMAX_HD void split_vehicle_fallback(ArenaState<NC>& s, const ArenaData& ad, StagedTick<NC>& work) {
    if (!work.independent) staged_vehicle<NC>(s, ad, work);
}
}
