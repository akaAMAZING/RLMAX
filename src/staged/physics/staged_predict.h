#pragma once
// Each body writes its own hit and rotation and returns flags for the arena reduction.
// Predictive rays are independent per body.
#include "../staged_tick.h"
namespace rlmax {
template <int NC>
RLMAX_HD uint32_t staged_predictive_lane(const ArenaData& ad, const Body* bodies, const CarGeo* geo,
                                         const V3* predict_ang, int body, PredictiveHit* hits, Q* rotations) {
    (void)predict_ang; (void)rotations;
    const bool hit = bodies[body].active && predictive_world(ad, bodies[body], body ? &geo[body - 1] : nullptr, hits[body]);
    return hit ? 1u : 0u;
}
template <int NC>
RLMAX_HD void staged_predictive_reduce(const uint32_t* flags, uint32_t& mask, uint32_t& reusable) {
    mask = reusable = 0;
    for (int body = 0; body <= NC; ++body) {
        if (flags[body] & 1u) mask |= 1u << body;
        if (flags[body] & 2u) reusable |= 1u << body;
    }
}
}
