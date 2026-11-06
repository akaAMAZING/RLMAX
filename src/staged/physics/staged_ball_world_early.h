#pragma once
// Ball-world remains one ordered manifold writer per arena.
#include "../staged_tick.h"
namespace rlmax {
template <int NC>
RLMAX_HD void staged_ball_world_early(ArenaState<NC>& s, const ArenaData& ad, const Body* bodies,
                                const TickConsts& k) {
    // The caller enters only for an already-awake ball.
    constexpr bool awake = true;
    auto& w = s.ball_wman;
#include "ball_world_body.inc"
}
}
