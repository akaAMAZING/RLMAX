#pragma once
// Per-body integration after solve; reduce the completed mask once per arena
// so simultaneous lanes never race on the shared seed word.
#include "../staged_tick.h"
namespace rlmax {
template <int NC>
RLMAX_HD bool staged_integrate_body(const ArenaState<NC>& s, Body* bodies, int body) {
    const bool active = body == 0 ? s.ball.sleeping != BALL_SLEEPING : bodies[body].active;
    if (!active) return false;
    Body& b = bodies[body];
    integrate_game(b, b.vel, b.ang, c::DT);
    return true;
}
template <int NC>
RLMAX_HD void staged_integrate_reduce(ArenaState<NC>& s, uint32_t integrated) {
    if (s.seed_mask) s.seed_mask &= ~(integrated | (integrated << 20));
}
}
