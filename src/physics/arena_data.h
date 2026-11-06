// Static arena data as the tick reads it (built on the host from the game's meshes; see crates/rlmax/src/arena.rs).
#pragma once

#include "math.h"

namespace rlmax {

// One triangle, 64 bytes, world space, Bullet units. Mirrors `Tri` in crates/rlmax/src/arena.rs.
struct Tri {
    V3 p0, p1, p2;
    V3 n;
    float ang[3];   // internal-edge angles v0v1, v1v2, v2v0 (0 = no neighbour: planar)
    uint32_t flags; // bits 0..5 edge flags; 6,7 goal origin; 8..11 mesh; 12..15 ray order; 16.. goal row or non-goal flags
};

enum TriFlag : uint32_t {
    TF_V0V1_CONVEX = 1u << 0,
    TF_V1V2_CONVEX = 1u << 1,
    TF_V2V0_CONVEX = 1u << 2,
    TF_V0V1_SWAP = 1u << 3,
    TF_V1V2_SWAP = 1u << 4,
    TF_V2V0_SWAP = 1u << 5,
    TF_ORIGIN_GOAL = 1u << 6,      // the mesh body sits at (0, +-GOAL_COMPONENT_Y_BT, 0)
    TF_ORIGIN_NEG = 1u << 7,       // ... the negative one
    TF_LEAF_CONTAINS = 1u << 16,   // non-goal only: the stored leaf box contains all three vertices
};

// the goal bodies' origin as the game holds it, two ulps under 102.4f (arena.rs GOAL_ORIGIN_Y)
constexpr float GOAL_COMPONENT_Y_BT = 102.39998626708984f;

// The world origin of the triangle's mesh body (its vertices in the body's frame are p - origin, as the loader built them).
RLMAX_HD V3 tri_origin(const Tri& t) {
    if (!(t.flags & TF_ORIGIN_GOAL)) return v3s(0.0f);
    return v3(0.0f, (t.flags & TF_ORIGIN_NEG) ? -GOAL_COMPONENT_Y_BT : GOAL_COMPONENT_Y_BT, 0.0f);
}

RLMAX_HD uint32_t tri_mesh(const Tri& t) { return (t.flags >> 8) & 0xfu; }
RLMAX_HD uint32_t tri_ray_rank(const Tri& t) { return (t.flags >> 12) & 0xfu; }

// A goal triangle's vertices in its body's frame: x and z as in the world, y the game's own local float (a goal body holds
// its vertices relative to its origin; the world y minus the origin has lost the low bits, arena.rs load_soccar_goal).
RLMAX_HD void goal_tri_local(const Tri& t, const float* goal_ly, V3 out[3]) {
    const float* y = goal_ly + 3u * (t.flags >> 16);
    out[0] = v3(t.p0.x, y[0], t.p0.z);
    out[1] = v3(t.p1.x, y[1], t.p1.z);
    out[2] = v3(t.p2.x, y[2], t.p2.z);
}

// The origin of mesh component m's body in the world: the goal components (4, 5 at +y, 6, 7 at -y) sit at the game's origin
RLMAX_HD V3 mesh_origin(uint32_t m) {
    if (m < 4 || m > 7) return v3s(0.0f);
    return v3(0.0f, m < 6 ? GOAL_COMPONENT_Y_BT : -GOAL_COMPONENT_Y_BT, 0.0f);
}

// Uniform grid: cell (x, y, z) lists cell_idx[cell_off[c] .. cell_off[c+1]] (triangles within `reach` of the cell).
struct ArenaData {
    const Tri* tris;
    const uint32_t* cell_off;  // num_cells + 1
    const uint16_t* cell_idx;
    const float* goal_ly;     // a goal triangle's local-frame vertex y, three per triangle, row flags >> 16
    const float* tri_box;     // each triangle's BVH leaf box in its body's frame, six floats; read only on a ray hit
    V3 grid_min;
    float inv_cell;
    int nx, ny, nz;
    uint32_t max_cell_count;  // the largest list, for fixed-bound loops
};

RLMAX_HD int cell_of(const ArenaData& a, V3 p) {
    int x = (int)floorf((p.x - a.grid_min.x) * a.inv_cell);
    int y = (int)floorf((p.y - a.grid_min.y) * a.inv_cell);
    int z = (int)floorf((p.z - a.grid_min.z) * a.inv_cell);
    x = x < 0 ? 0 : (x >= a.nx ? a.nx - 1 : x);
    y = y < 0 ? 0 : (y >= a.ny ? a.ny - 1 : y);
    z = z < 0 ? 0 : (z >= a.nz ? a.nz - 1 : z);
    return (z * a.ny + y) * a.nx + x;
}

}  // namespace rlmax
