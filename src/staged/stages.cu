// Staged small-batch image, built separately from the ordinary tick.
#include "staged_tick.h"
#include "physics/staged_ball_world_early.h"
#include "kernels/staged_integrate_post.cuh"
using namespace rlmax;
// Small-batch staging. Each arena's
// scratch is private; interacting vehicles retain the original visit order.
template <int NC>
__device__ __forceinline__ void staged_control(uint32_t* states, uint32_t n, const uint8_t* rows,
                                                const float* table, uint32_t* goals) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ArenaState<NC>& a = reinterpret_cast<ArenaState<NC>*>(states)[i];
    if (a.events & 3u) { kickoff_reset<NC>(a); goals[i] += 1; }
    for (int c = 0; c < NC; ++c) {
        const float* t = table + 8 * (rows ? (uint32_t)rows[i * NC + c] : i * NC + c);
        Controls& ctl = a.cars[c].ctl;
        ctl.throttle = t[0]; ctl.steer = t[1]; ctl.pitch = t[2]; ctl.yaw = t[3]; ctl.roll = t[4];
        ctl.buttons = (t[5] != 0.0f ? BTN_JUMP : 0u) | (t[6] != 0.0f ? BTN_BOOST : 0u) |
                      (t[7] != 0.0f ? BTN_HANDBRAKE : 0u);
    }
}
#define RLMAX_STAGED_KERNELS(N) \
extern "C" __global__ void rlmax_staged_controls_##N(uint32_t* states, uint32_t n, const uint8_t* rows, const float* table, uint32_t* goals) { \
    staged_control<N>(states, n, rows, table, goals); \
} \
extern "C" __global__ void rlmax_staged_begin_##N(uint32_t* states, uint32_t n, ArenaData ad, StagedTick<N>* work) { \
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) staged_begin<N>(reinterpret_cast<ArenaState<N>*>(states)[i], ad, work[i]); \
} \
extern "C" __global__ void rlmax_staged_pre_cars_##N(uint32_t* states, uint32_t n, ArenaData ad, StagedTick<N>* work) { \
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x, i = t / N, c = t % N; \
    if (i < n) staged_pre_car<N>(c, reinterpret_cast<ArenaState<N>*>(states)[i], ad, work[i]); \
} \
extern "C" __global__ void rlmax_staged_early_ball_##N(uint32_t* states, uint32_t n, ArenaData ad, StagedTick<N>* work) { \
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n && work[i].bodies[0].active) \
        staged_ball_world_early<N>(reinterpret_cast<ArenaState<N>*>(states)[i], ad, work[i].bodies, tick_consts()); \
} \
extern "C" __global__ void rlmax_staged_pre_integrate_##N(uint32_t* states, uint32_t n, ArenaData ad, StagedTick<N>* work) { \
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) staged_finish_pre_integrate<N>(reinterpret_cast<ArenaState<N>*>(states)[i], ad, work[i].epa[0], work[i]); \
} \
extern "C" __global__ void rlmax_staged_integrate_post_##N(uint32_t* states, uint32_t n, ArenaData ad, StagedTick<N>* work) { \
    staged_integrate_post<N>(n, ad, reinterpret_cast<ArenaState<N>*>(states), work); \
}
RLMAX_STAGED_KERNELS(2)
RLMAX_STAGED_KERNELS(4)
RLMAX_STAGED_KERNELS(6)
#undef RLMAX_STAGED_KERNELS
#include "kernels/staged_body_split.cu"
