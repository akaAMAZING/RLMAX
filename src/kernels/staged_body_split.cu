// Split of the staged body phase. Each kernel has one kind of
// work; the next node sees all writes from the previous node. The ordinary
// module remains unchanged. Scalar contact and sweep physics is shared.
#include "physics/tick.h"
#include "physics/staged_car_ball.h"
#include "physics/staged_predict.h"
#include "physics/staged_vehicle_split.h"
#include "staged_wheels_split.cuh"
#include "world_lanes.cuh"
using namespace rlmax;

template<int NC> __device__ __forceinline__ void split_ball(uint32_t* states, uint32_t n, StagedTick<NC>* work) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x, arena = t / NC, car = t % NC;
    if (arena >= n) return;
    auto& s = reinterpret_cast<ArenaState<NC>*>(states)[arena]; auto& w = work[arena];
    staged_car_ball<NC>(car, s, w.bodies, w.geo.g, tick_consts(), w.ball_stage_snap,
                        w.ball_stage_points, w.ball_stage_counts, w.ball_stage_touch);
}
// Slots are appended after the array of StagedTick records. Each body owns
// one slot; the next graph node reconstructs masks without a warp ballot.
template<int NC> __device__ __forceinline__ uint32_t* split_flags(StagedTick<NC>* work, uint32_t n) {
    return reinterpret_cast<uint32_t*>(work + n);
}
template<int NC> __device__ __forceinline__ void split_predict(uint32_t n, ArenaData ad, StagedTick<NC>* work) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x, arena = t / (NC + 1), body = t % (NC + 1);
    if (arena >= n) return;
    auto& w = work[arena];
    split_flags<NC>(work, n)[t] = staged_predictive_lane<NC>(ad, w.bodies, w.geo.g, w.predict_ang,
                                                             body, w.predictive, w.sweep_rotations);
}
template<int NC> __device__ __forceinline__ void split_sweep(uint32_t* states, uint32_t n, StagedTick<NC>* work) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x, arena = t / (NC + 1), body = t % (NC + 1);
    if (arena >= n) return;
    auto& s = reinterpret_cast<ArenaState<NC>*>(states)[arena]; auto& w = work[arena];
    const uint32_t* flags = split_flags<NC>(work, n) + arena * (NC + 1);
    if (body == 0) staged_predictive_reduce<NC>(flags, w.predictive_mask, w.reusable_rotations);
    sweep_prepare_body<NC>(s.sweep, w.prepared_sweep, w.bodies, w.predict_lin, w.predict_ang,
                           w.geo.g, tick_consts().ball_radius, w.sweep_rotations,
                           (flags[body] & 2u) ? (1u << body) : 0u, body);
}
template<int NC> __device__ __forceinline__ void split_gate(uint32_t* states, uint32_t n, StagedTick<NC>* work) {
    const uint32_t arena = blockIdx.x * blockDim.x + threadIdx.x;
    if (arena < n) split_vehicle_gate<NC>(reinterpret_cast<ArenaState<NC>*>(states)[arena], work[arena]);
}
template<int NC> __device__ __forceinline__ void split_fallback(uint32_t* states, uint32_t n, ArenaData ad, StagedTick<NC>* work) {
    const uint32_t arena = blockIdx.x * blockDim.x + threadIdx.x;
    if (arena < n) split_vehicle_fallback<NC>(reinterpret_cast<ArenaState<NC>*>(states)[arena], ad, work[arena]);
}
#define SPLIT_KERNELS(N) \
extern "C" __global__ void rlmax_vehicle_wheels_##N(uint32_t* s, uint32_t n, ArenaData ad, StagedTick<N>* work) { \
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x, arena = t / (4 * N), car = (t / 4) % N, wheel = t % 4; \
    if (arena < n) split_vehicle_car_lanes<N>(car, wheel, reinterpret_cast<ArenaState<N>*>(s)[arena], ad, work[arena]); \
} \
extern "C" __global__ void rlmax_vehicle_gate_##N(uint32_t* s, uint32_t n, ArenaData, StagedTick<N>* w) { split_gate<N>(s,n,w); } \
extern "C" __global__ void rlmax_vehicle_fallback_##N(uint32_t* s, uint32_t n, ArenaData ad, StagedTick<N>* w) { split_fallback<N>(s,n,ad,w); } \
extern "C" __global__ void rlmax_body_world_##N(uint32_t* s, uint32_t n, ArenaData ad, StagedTick<N>* w) { world_lanes<N>(s,n,ad,w); } \
extern "C" __global__ void rlmax_body_ball_##N(uint32_t* s, uint32_t n, ArenaData, StagedTick<N>* w) { split_ball<N>(s,n,w); } \
extern "C" __global__ void rlmax_body_predict_##N(uint32_t*, uint32_t n, ArenaData ad, StagedTick<N>* w) { split_predict<N>(n,ad,w); } \
extern "C" __global__ void rlmax_body_sweep_##N(uint32_t* s, uint32_t n, ArenaData, StagedTick<N>* w) { split_sweep<N>(s,n,w); }
SPLIT_KERNELS(2)
SPLIT_KERNELS(4)
SPLIT_KERNELS(6)
#undef SPLIT_KERNELS
