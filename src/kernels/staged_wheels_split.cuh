#pragma once
#include "physics/tick.h"
using namespace rlmax;
// Four adjacent lanes own one car's wheels. Each phase reads one body state;
// lane zero applies the four velocity deltas in the game's wheel order.
// Ordered wheel impulses and contacts retain the existing RLMAX arithmetic.
__device__ __forceinline__ void split_combine_wheels(Body& rb, WheelDv delta, unsigned mask, unsigned lane) {
    V3 vel = v3s(0.0f), ang = v3s(0.0f);
    if (lane == 0) { vel = rb.vel; ang = rb.ang; }
    for (int q = 0; q < 4; ++q) {
        const int applied = __shfl_sync(mask, (int)delta.applied, q, 4);
        const V3 dv = v3(__shfl_sync(mask, delta.dv.x, q, 4), __shfl_sync(mask, delta.dv.y, q, 4),
                         __shfl_sync(mask, delta.dv.z, q, 4));
        const V3 dw = v3(__shfl_sync(mask, delta.dw.x, q, 4), __shfl_sync(mask, delta.dw.y, q, 4),
                         __shfl_sync(mask, delta.dw.z, q, 4));
        if (lane == 0 && applied) { vel += dv; ang += dw; }
    }
    if (lane == 0) { rb.vel = vel; rb.ang = ang; }
    __syncwarp(mask);
}

template <int NC>
__device__ __forceinline__ void split_vehicle_car_lanes(int i, int wheel, ArenaState<NC>& s, const ArenaData& ad, StagedTick<NC>& work) {
    if (!work.independent || !work.pre_ticked[i]) return;
    const unsigned warp_lane = threadIdx.x & 31u;
    const unsigned lane = warp_lane & 3u;
    const unsigned mask = 15u << (warp_lane & ~3u);
    Body& rb = work.bodies[1 + i];
    const StagedWheel local_wheel = staged_vehicle_wheel<NC>(i, wheel, s, ad, work);
    const unsigned hit_mask = __ballot_sync(mask, local_wheel.hit) >> (warp_lane & ~3u);
    const int num_contact = __popc(hit_mask);
    if (num_contact) {
        WheelDv delta = staged_suspension_wheel(rb, work.geo.g[i], local_wheel, wheel);
        split_combine_wheels(rb, delta, mask, lane);
        delta = staged_friction_wheel(rb, local_wheel, work.outs[i], num_contact);
        split_combine_wheels(rb, delta, mask, lane);
    }
    uint32_t bits = 0;
    V3 nsum = v3s(0.0f);
    for (int q = 0; q < 4; ++q) {
        const bool hit = (hit_mask & (1u << q)) != 0;
        const V3 normal = v3(__shfl_sync(mask, local_wheel.contact_normal.x, q, 4),
                             __shfl_sync(mask, local_wheel.contact_normal.y, q, 4),
                             __shfl_sync(mask, local_wheel.contact_normal.z, q, 4));
        if (lane == 0 && hit) { bits |= (1u << (16 + q)) | (1u << (20 + q)); nsum += normal; }
    }
    if (lane == 0) {
        CarState& cs = s.cars[i];
        cs.flags = (cs.flags & ~(0xFFu << 16)) | bits;
        set_fl(cs, CF_ON_GROUND, wheel_count(cs.flags) >= 3);
        set_fl(cs, CF_STICKY_GATE, num_contact != 0);
        cs.wheel_normal_sum = nsum;
    }
}
