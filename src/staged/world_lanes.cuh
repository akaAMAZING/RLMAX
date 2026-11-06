#pragma once
#include "physics/world_lane_helpers.h"
using namespace rlmax;

// Four adjacent lanes own one car. They scan disjoint contiguous cell ranges
// and retain hits locally; lane zero inserts them in the original cell order.
// Any local overflow reruns the entire original mesh loop for that car.
template<int NC> __device__ __forceinline__ void world_lanes(uint32_t* states,uint32_t n,ArenaData ad,StagedTick<NC>* work) {
    constexpr unsigned lanes=4;
    const uint32_t t=blockIdx.x*blockDim.x+threadIdx.x, owner=t/lanes;
    const uint32_t arena=owner/NC, car=owner%NC;
    if (arena>=n) return;
    const unsigned lane=threadIdx.x&3u;
    const unsigned mask=15u << (threadIdx.x&28u);
    auto& s=reinterpret_cast<ArenaState<NC>*>(states)[arena]; auto& w=work[arena];
    if (lane==0) world_lane_prefix<NC>(car,s,ad,w.bodies,w.geo.g,w.world_stage_points,w.world_stage_counts,
                                       w.split_car_world_hit,w.split_car_world_normal);
    const Body& rb=w.bodies[1+car];
    uint32_t start=0,end=0;
    if (rb.contact_response) {
        const Obb ob=car_obb(rb,w.geo.g[car]); const int cell=cell_of(ad,ob.c);
        start=ad.cell_off[cell]; end=ad.cell_off[cell+1];
    }
    const uint64_t length=uint64_t(end)-start;
    const uint32_t first=start+uint32_t(length*lane/lanes);
    const uint32_t last=start+uint32_t(length*(lane+1)/lanes);
    WorldLaneHit hits[WORLD_LANE_HIT_CAP]; int count=0; EpaScratch& scratch=w.epa[car*lanes+lane];
    const bool overflow=world_lane_scan<NC>(car,ad,w.bodies,w.geo.g,first,last,hits,count,scratch);
    if (__any_sync(mask,overflow)) {
        if (lane==0) world_lane_fallback<NC>(car,s,ad,w.bodies,w.geo.g,w.world_stage_points,w.world_stage_counts,
                                            w.split_car_world_hit,w.split_car_world_normal,scratch);
    } else {
        for (int source=0;source<int(lanes);++source) {
            const int source_count=__shfl_sync(mask,count,source,lanes);
            for (int j=0;j<source_count;++j) {
                WorldLaneHit own{}; if (j<count) own=hits[j];
                WorldLaneHit out;
                out.mesh=__shfl_sync(mask,own.mesh,source,lanes);
                out.point.la.x=__shfl_sync(mask,own.point.la.x,source,lanes);
                out.point.la.y=__shfl_sync(mask,own.point.la.y,source,lanes);
                out.point.la.z=__shfl_sync(mask,own.point.la.z,source,lanes);
                out.point.lb.x=__shfl_sync(mask,own.point.lb.x,source,lanes);
                out.point.lb.y=__shfl_sync(mask,own.point.lb.y,source,lanes);
                out.point.lb.z=__shfl_sync(mask,own.point.lb.z,source,lanes);
                out.point.n.x=__shfl_sync(mask,own.point.n.x,source,lanes);
                out.point.n.y=__shfl_sync(mask,own.point.n.y,source,lanes);
                out.point.n.z=__shfl_sync(mask,own.point.n.z,source,lanes);
                out.point.d=__shfl_sync(mask,own.point.d,source,lanes);
                if (lane==0) if (Manifold* m=wman_get(s.cars[car].wman,out.mesh))
                    man_add_point(*m,out.point,w.geo.g[car].contact_threshold);
            }
        }
    }
    if (lane==0) world_lane_finish<NC>(car,s,ad,w.bodies,w.geo.g,w.world_stage_points,w.world_stage_counts,
                                      w.split_car_world_hit,w.split_car_world_normal);
}
