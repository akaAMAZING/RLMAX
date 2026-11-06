#pragma once
#include "physics/tick.h"
#include "physics/staged_integrate.h"

namespace rlmax {
// Run independent body integration in a warp, then let each arena leader
// perform ordered callbacks after all body writes are visible.
template <int NC>
__device__ __forceinline__ void staged_integrate_post(uint32_t n, ArenaData ad,
                                                      ArenaState<NC>* states, StagedTick<NC>* work) {
    constexpr int lanes = NC <= 2 ? 4 : 8;
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t arena = t / lanes;
    const int body = int(t % lanes);
    const bool valid = arena < n && body <= NC;
    const bool integrated = valid && staged_integrate_body<NC>(states[arena], work[arena].bodies, body);
    const unsigned votes = __ballot_sync(0xffffffffu, integrated);
    if (valid && body == 0) {
        constexpr unsigned used = (1u << (NC + 1)) - 1u;
        staged_integrate_reduce<NC>(states[arena], (votes >> (threadIdx.x & 31u)) & used);
    }
    // All lanes, including padding, participate. Leaders then read the other
    // lanes' body writes in the ordered post-step.
    __syncwarp();
    if (valid && body == 0)
        staged_finish_callbacks<NC>(states[arena], ad, work[arena].epa[0], work[arena]);
    __syncwarp();
    if (valid && body > 0) staged_car_post<NC>(body - 1, states[arena], work[arena]);
    __syncwarp();
    if (valid && body == 0)
        staged_finish_final<NC>(states[arena], ad, work[arena].epa[0], work[arena]);
}
}  // namespace rlmax
