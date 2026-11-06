//! Arena static data: the game's collision mesh, made uniform (DESIGN §4.2).
//!
//! Built at start-up from the `.cmf` files in `$RLMAX_DATA/meshes/soccar` (never in
//! the repo). Every triangle keeps its exact vertices plus the internal-edge data the contact code needs. A uniform
//! grid over the arena lists, per cell, every triangle within `REACH` of the cell, padded to a fixed `TRIS_PER_CELL`
//! with a sentinel triangle that can never collide, so every query is one cell read and a fixed number of tests.

use std::path::Path;

/// Bullet units per Unreal unit.
pub const UU_TO_BT: f32 = 1.0 / 50.0;

/// One triangle as the device reads it: 64 bytes. Must match `Tri` in `src/physics/arena_data.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tri {
    pub p: [[f32; 3]; 3],
    pub n: [f32; 3],
    /// Internal-edge angles for edges v0v1, v1v2, v2v0 (0 = no neighbour: a free edge is planar).
    pub ang: [f32; 3],
    /// Bits 0..=5: Bullet's triangle-info flags (convex, swap-normal per edge). Bits 6, 7: goal origin. Bits 8..=11: mesh
    /// index. Bits 12..=15: the mesh's place in the game's ray order. For goals, bits 16.. hold the local-y row;
    /// for other triangles bit 16 asserts that the stored leaf box encloses every vertex.
    pub flags: u32,
}

pub const FLAG_V0V1_CONVEX: u32 = 1;
pub const FLAG_V1V2_CONVEX: u32 = 1 << 1;
pub const FLAG_V2V0_CONVEX: u32 = 1 << 2;
pub const FLAG_V0V1_SWAP_NORMAL_B: u32 = 1 << 3;
pub const FLAG_V1V2_SWAP_NORMAL_B: u32 = 1 << 4;
pub const FLAG_V2V0_SWAP_NORMAL_B: u32 = 1 << 5;
/// Non-goal triangle only: its stored BVH leaf box contains all three vertices.
pub const FLAG_LEAF_CONTAINS: u32 = 1 << 16;

const TAU: f32 = std::f32::consts::TAU;
const EQUAL_VERTEX_THRESHOLD: f32 = 0.0001 * 0.0001;
const PLANAR_EPSILON: f32 = 0.0001;

type V = [f32; 3];
fn sub(a: V, b: V) -> V { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn add(a: V, b: V) -> V { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn scale(a: V, s: f32) -> V { [a[0] * s, a[1] * s, a[2] * s] }
fn dot(a: V, b: V) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: V, b: V) -> V { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
fn len2(a: V) -> f32 { dot(a, a) }
fn normalize(a: V) -> V { scale(a, 1.0 / len2(a).sqrt()) }

/// Rotate `v` about unit `axis` by `angle` (Rodrigues; offline only).
fn rotate(v: V, axis: V, angle: f32) -> V {
    // Through f64: f32::sin_cos and f32::atan2 come from the platform's libm, and glibc and UCRT differ in the last bit.
    let (s, c) = { let (s, c) = (angle as f64).sin_cos(); (s as f32, c as f32) };
    add(add(scale(v, c), scale(cross(axis, v), s)), scale(axis, dot(axis, v) * (1.0 - c)))
}

/// A `.cmf` file: `u32 num_tris, u32 num_verts, u32 indices[3·tris], f32 verts[3·verts]`, little-endian, Bullet units.
pub fn read_cmf(bytes: &[u8]) -> Result<Vec<[V; 3]>, String> {
    let u32_at = |i: usize| -> Result<u32, String> {
        bytes.get(i..i + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).ok_or_else(|| "truncated cmf".to_owned())
    };
    let nt = u32_at(0)? as usize;
    let nv = u32_at(4)? as usize;
    if nt == 0 || nv == 0 || nt > 1_000_000 || nv > 1_000_000 {
        return Err(format!("bad cmf counts {nt}/{nv}"));
    }
    let vbase = 8 + nt * 12;
    let f = |i: usize| -> Result<f32, String> { Ok(f32::from_bits(u32_at(vbase + i * 4)?)) };
    let mut verts = Vec::with_capacity(nv);
    for i in 0..nv {
        verts.push([f(i * 3)?, f(i * 3 + 1)?, f(i * 3 + 2)?]);
    }
    let mut tris = Vec::with_capacity(nt);
    for t in 0..nt {
        let idx = |k: usize| -> Result<V, String> {
            let vi = u32_at(8 + (t * 3 + k) * 4)? as usize;
            verts.get(vi).copied().ok_or_else(|| format!("bad vertex index {vi}"))
        };
        tris.push([idx(0)?, idx(1)?, idx(2)?]);
    }
    Ok(tris)
}

fn tri_normal(p: &[V; 3]) -> V {
    // Bullet: (p1 - p0) x (p2 - p0), normalised.
    normalize(cross(sub(p[1], p[0]), sub(p[2], p[0])))
}

fn tri_normal_len2(p: &[V; 3]) -> f32 { len2(cross(sub(p[1], p[0]), sub(p[2], p[0]))) }

/// Bullet's internal-edge connectivity (`btGenerateInternalEdgeInfo`), per mesh: for every edge shared with another
/// triangle of the same mesh, the signed angle between the faces and whether the edge is convex.
fn internal_edge_info(mesh: &[[V; 3]]) -> Vec<([f32; 3], u32)> {
    let mut info = vec![([TAU; 3], 0u32); mesh.len()];
    let aabbs: Vec<(V, V)> = mesh
        .iter()
        .map(|t| {
            let lo = [0, 1, 2].map(|k| t[0][k].min(t[1][k]).min(t[2][k]));
            let hi = [0, 1, 2].map(|k| t[0][k].max(t[1][k]).max(t[2][k]));
            (lo, hi)
        })
        .collect();
    for (ia, a) in mesh.iter().enumerate() {
        if tri_normal_len2(a) < EQUAL_VERTEX_THRESHOLD {
            continue;
        }
        let na = tri_normal(a);
        for (ib, b) in mesh.iter().enumerate() {
            if ia == ib || tri_normal_len2(b) < EQUAL_VERTEX_THRESHOLD {
                continue;
            }
            let (alo, ahi) = aabbs[ia];
            let (blo, bhi) = aabbs[ib];
            if (0..3).any(|k| alo[k] > bhi[k] || blo[k] > ahi[k]) {
                continue;
            }
            let mut shared_a = [0usize; 2];
            let mut shared_b = [0usize; 2];
            let mut num = 0;
            for (i, &va) in a.iter().enumerate() {
                for (j, &vb) in b.iter().enumerate() {
                    if len2(sub(va, vb)) < EQUAL_VERTEX_THRESHOLD && num < 2 {
                        shared_a[num] = i;
                        shared_b[num] = j;
                        num += 1;
                    }
                }
            }
            if num != 2 {
                continue;
            }
            if shared_a[0] == 0 && shared_a[1] == 2 {
                shared_a = [2, 0];
                shared_b.swap(0, 1);
            }
            let sum_a = shared_a[0] + shared_a[1];
            let other_a = 3 - sum_a;
            let edge = normalize(sub(a[shared_a[1]], a[shared_a[0]]));
            let other_b = 3 - (shared_b[0] + shared_b[1]);
            let nb = tri_normal(&[b[shared_b[1]], b[shared_b[0]], b[other_b]]);

            let mut ec_a = normalize(cross(edge, na));
            if dot(ec_a, sub(a[other_a], a[shared_a[0]])) < 0.0 {
                ec_a = scale(ec_a, -1.0);
            }
            let mut ec_b = normalize(cross(edge, nb));
            if dot(ec_b, sub(b[other_b], b[shared_b[0]])) < 0.0 {
                ec_b = scale(ec_b, -1.0);
            }
            let calc_edge = cross(ec_a, ec_b);
            let mut corrected = 0.0f32;
            let mut convex = false;
            if len2(calc_edge) >= PLANAR_EPSILON {
                let calc_edge = normalize(calc_edge);
                let calc_na = normalize(cross(calc_edge, ec_a));
                // get_angle(edge_a, normal_a, normal_b) = atan2(normal_b . edge_a, normal_b . normal_a)
                let angle2 = (dot(ec_b, calc_na) as f64).atan2(dot(ec_b, ec_a) as f64) as f32;
                let ang4 = std::f32::consts::PI - angle2;
                convex = dot(na, ec_b) < 0.0;
                corrected = if convex { ang4 } else { -ang4 };
            }
            // Edge index by the sum of the shared vertex indices: 1 -> v0v1, 3 -> v1v2, 2 -> v2v0.
            let (slot, edge_vec, convex_bit, swap_bit) = match sum_a {
                1 => (0, sub(a[1], a[0]), FLAG_V0V1_CONVEX, FLAG_V0V1_SWAP_NORMAL_B),
                3 => (1, sub(a[2], a[1]), FLAG_V1V2_CONVEX, FLAG_V1V2_SWAP_NORMAL_B),
                _ => (2, sub(a[0], a[2]), FLAG_V2V0_CONVEX, FLAG_V2V0_SWAP_NORMAL_B),
            };
            let e = normalize(scale(edge_vec, -1.0));
            let computed_nb = rotate(na, e, -corrected);
            let entry = &mut info[ia];
            if dot(computed_nb, nb) < 0.0 {
                entry.1 |= swap_bit;
            }
            entry.0[slot] = -corrected;
            if convex {
                entry.1 |= convex_bit;
            }
        }
    }
    // an edge with no neighbouring triangle is planar: angle 0, not 2π
    for (ang, _) in &mut info {
        for a in ang.iter_mut().filter(|a| **a == TAU) {
            *a = 0.0;
        }
    }
    info
}

/// The mesh identity (over the index list and vertex positions as stored). Offline only.
fn cmf_hash(bytes: &[u8]) -> u32 {
    let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    let (nt, nv) = (u32_at(0) as usize, u32_at(4) as usize);
    let vbase = 8 + nt * 12;
    let mut hash = (nv + nt * nv) as u32;
    for i in 0..nt * 3 {
        let vi = u32_at(8 + i * 4) as usize;
        for k in 0..3 {
            let pos = f32::from_bits(u32_at(vbase + (vi * 3 + k) * 4));
            let mut v = pos as i32 as u32;
            v = ((v >> 16) ^ v).wrapping_mul(0x045D_9F3B);
            v = ((v >> 16) ^ v).wrapping_mul(0x045D_9F3B);
            v ^= v >> 16;
            hash ^= v.wrapping_add(0x9E37_79B9).wrapping_add(hash << 6).wrapping_add(hash >> 2);
        }
    }
    hash
}

/// The game's Soccar components in their native order (the order the arena adds them, and so the order Bullet pairs
/// and solves them), by mesh hash.
const SOCCAR_COMPONENT_ORDER: [u32; 16] = [
    0xA160_BAF9, 0x2811_EEE8, 0x14B8_4668, 0xEC75_9EBF, 0x94FB_0D5C, 0xDEA0_7102, 0xBD4F_BEA8, 0x39A4_7F63,
    0x3D79_D25D, 0xD84C_7A68, 0xB81A_C8B9, 0x7603_58D3, 0x73AE_4940, 0x918F_4A4E, 0x1F8E_E550, 0x255B_A8C1,
];

/// The goal components are placed at y = ±102.4 BT and hold their triangles relative to that origin; every other
/// component sits at the world origin. Chosen by geometry (vertex-AABB centre beyond the back wall).
const GOAL_COMPONENT_Y: f32 = 102.4;
/// The goal bodies' real origin: the game's float, two ulps under 102.4f (the mesh audit finds the
/// same value in every goal vertex). Must match GOAL_COMPONENT_Y_BT in `src/physics/arena_data.h`.
#[allow(clippy::excessive_precision)] // the game's float, written out exactly
const GOAL_ORIGIN_Y: f32 = 102.399_986_267_089_84;

fn component_translation(mesh: &[[V; 3]]) -> V {
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for t in mesh {
        for p in t {
            lo = lo.min(p[1]);
            hi = hi.max(p[1]);
        }
    }
    let cy = (lo + hi) * 0.5;
    if cy > GOAL_COMPONENT_Y {
        [0.0, GOAL_ORIGIN_Y, 0.0]
    } else if cy < -GOAL_COMPONENT_Y {
        [0.0, -GOAL_ORIGIN_Y, 0.0]
    } else {
        [0.0; 3]
    }
}

/// Bullet's 16-bit AABB quantisation over a mesh's local AABB (btQuantizedBvh::setQuantizationValues, margin 1).
struct Quant {
    min: V,
    scale: V,
}

impl Quant {
    const RANGE: f32 = 65533.0;

    fn new(lo: V, hi: V) -> Self {
        let mut min = sub(lo, [1.0; 3]);
        let mut max = add(hi, [1.0; 3]);
        let scale_of = |min: V, max: V| [0, 1, 2].map(|k| Self::RANGE / (max[k] - min[k]));
        let mut q = Quant { min, scale: scale_of(min, max) };
        // the refinement pass (scale kept between the two steps)
        let umin = q.unquantize(q.quantize(min, false));
        min = [0, 1, 2].map(|k| min[k].min(umin[k] - 1.0));
        q.min = min;
        let umax = q.unquantize(q.quantize(max, true));
        max = [0, 1, 2].map(|k| max[k].max(umax[k] + 1.0));
        q.min = min;
        q.scale = scale_of(min, max);
        q
    }

    fn quantize(&self, p: V, is_max: bool) -> [u16; 3] {
        [0, 1, 2].map(|k| {
            let v = (p[k] - self.min[k]) * self.scale[k];
            if is_max { ((v + 1.0) as i32 as u16) | 1 } else { (v as i32 as u16) & 0xfffe }
        })
    }

    fn unquantize(&self, q: [u16; 3]) -> V {
        [0, 1, 2].map(|k| q[k] as f32 / self.scale[k] + self.min[k])
    }
}

/// The order in which Bullet's quantized BVH (btOptimizedBvh::build, btQuantizedBvh::buildTree) visits a mesh's
/// triangles: its stackless walk reports overlapping leaves left to right, and the leaves end where the build's in-place
/// partitions put them. Contacts reach a persistent manifold in this order, and which 4 points a full manifold keeps
/// (sortCachedPoints) depends on it, as does which neighbour an edge's info comes from. Returns triangle indices.
fn bullet_leaf_order(mesh: &[[V; 3]]) -> Vec<usize> {
    let q = mesh_quant(mesh);
    // leaf centres from the quantised leaf AABBs, as getAabbMin/Max read them back
    let mut leaves: Vec<(V, usize)> = mesh
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let (ua, ub) = leaf_box(&q, t);
            (scale(add(ub, ua), 0.5), i)
        })
        .collect();
    bullet_partition(&mut leaves);
    leaves.into_iter().map(|l| l.1).collect()
}

/// The mesh's BVH quantisation, from its local AABB: the vertex extremes (btTriangleMeshShape::recalcLocalAabb, concave margin 0).
fn mesh_quant(mesh: &[[V; 3]]) -> Quant {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for t in mesh {
        for p in t {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    Quant::new(lo, hi)
}

/// A triangle's BVH leaf box as the walk reads it back: its AABB, widened to the minimum size, quantised outwards.
fn leaf_box(q: &Quant, t: &[V; 3]) -> (V, V) {
    const MIN_AABB_DIMENSION: f32 = 0.002;
    const MIN_AABB_HALF_DIMENSION: f32 = MIN_AABB_DIMENSION / 2.0;
    let mut a = [0, 1, 2].map(|k| t[0][k].min(t[1][k]).min(t[2][k]));
    let mut b = [0, 1, 2].map(|k| t[0][k].max(t[1][k]).max(t[2][k]));
    for k in 0..3 {
        if b[k] - a[k] < MIN_AABB_DIMENSION {
            b[k] += MIN_AABB_HALF_DIMENSION;
            a[k] -= MIN_AABB_HALF_DIMENSION;
        }
    }
    (q.unquantize(q.quantize(a, false)), q.unquantize(q.quantize(b, true)))
}

/// Each mesh's place in the order the game's ray visits them: by the broadphase's quantised top x of the mesh box, ties by mesh. A ray
/// hitting two meshes at the same fraction, as at the mirror seam at x = 0, keeps the one visited first.
fn mesh_ray_ranks<'a>(meshes: impl Iterator<Item = &'a Vec<[V; 3]>>) -> Vec<u32> {
    #[allow(clippy::excessive_precision)] // the game's float, written out exactly
    const SWEEP_HALF_BT: f32 = 10485.740234375; // the game's broadphase half extent
    let key: Vec<i32> = meshes
        .map(|m| {
            let top = m.iter().flatten().fold(f32::MIN, |a, p| a.max(p[0]));
            let scaled = (top + SWEEP_HALF_BT) * (65535.0 / (SWEEP_HALF_BT + SWEEP_HALF_BT));
            (scaled.clamp(0.0, 65535.0) as i32 & 0xfffe) | 1
        })
        .collect();
    let mut order: Vec<usize> = (0..key.len()).collect();
    order.sort_by_key(|&i| key[i]);
    let mut rank = vec![0; key.len()];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r as u32;
    }
    rank
}

/// btQuantizedBvh::calcSplittingAxis + sortAndCalcSplittingIndex, recursively.
fn bullet_partition(leaves: &mut [(V, usize)]) {
    let n = leaves.len();
    if n < 2 {
        return;
    }
    let mut mean = [0.0f32; 3];
    for l in leaves.iter() {
        mean = add(mean, l.0);
    }
    mean = scale(mean, 1.0 / n as f32);
    let mut var = [0.0f32; 3];
    for l in leaves.iter() {
        let d = sub(l.0, mean);
        var = add(var, [d[0] * d[0], d[1] * d[1], d[2] * d[2]]);
    }
    var = scale(var, 1.0 / (n as f32 - 1.0));
    let axis = if var[0] < var[1] { if var[1] < var[2] { 2 } else { 1 } } else if var[0] < var[2] { 2 } else { 0 };
    let split_value = mean[axis];
    let mut s = 0;
    for i in 0..n {
        if leaves[i].0[axis] > split_value {
            leaves.swap(i, s);
            s += 1;
        }
    }
    let balanced = n / 3;
    if s <= balanced || s >= n - 1 - balanced {
        s = n >> 1;
    }
    let (left, right) = leaves.split_at_mut(s);
    bullet_partition(left);
    bullet_partition(right);
}

/// Every Soccar triangle, in Bullet units and world space, with internal-edge data. Meshes come in the game's native
/// component order and each mesh's triangles in Bullet's BVH visiting order, so a cell list sorted by index is the order
/// in which Bullet would process the same triangles. Edge data is computed in each component's local frame, as Bullet's
/// btGenerateInternalEdgeInfo does on the mesh it is given.
pub fn load_soccar(mesh_dir: &Path) -> Result<Vec<Tri>, String> {
    load_soccar_goal(mesh_dir).map(|(tris, _, _)| tris)
}

type GoalMeshes = (Vec<Tri>, Vec<f32>, Vec<f32>);
type GoalMesh = (usize, Vec<[V; 3]>, Option<Vec<[V; 3]>>);

/// The game's local-frame y of each goal triangle's three vertices (the goal bodies hold their vertices relative to their
/// origin, and the game's local values are not the world values minus 102.4f: the world float has lost the low bits). The
/// table is read from `mesh_N.local` beside `mesh_N.cmf` (f32 xyz per cmf vertex, derived from the game's package, never in
/// the repo); a goal component without one gets world minus origin. Triangle `flags >> 16` is the row.
pub fn load_soccar_goal(mesh_dir: &Path) -> Result<GoalMeshes, String> {
    let files: Vec<_> = std::fs::read_dir(mesh_dir)
        .map_err(|e| format!("{}: {e}", mesh_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "cmf"))
        .collect();
    if files.len() != 16 {
        return Err(format!("expected 16 Soccar meshes in {}, found {}", mesh_dir.display(), files.len()));
    }
    // (component order, world triangles, the game's local triangles when its .local file is there)
    let mut meshes: Vec<GoalMesh> = Vec::with_capacity(16);
    for path in &files {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mesh = read_cmf(&bytes)?;
        let hash = cmf_hash(&bytes);
        let order = SOCCAR_COMPONENT_ORDER
            .iter()
            .position(|&h| h == hash)
            .ok_or_else(|| format!("{}: not a Soccar component (hash {hash:#010x})", path.display()))?;
        let game_local = match std::fs::read(path.with_extension("local")) {
            Ok(lb) => {
                let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
                let (nt, nv) = (u32_at(0), u32_at(4));
                if lb.len() != nv * 12 {
                    return Err(format!("{}: {} bytes for {nv} vertices", path.with_extension("local").display(), lb.len()));
                }
                let vert = |i: usize| [0, 1, 2].map(|k| f32::from_le_bytes(lb[i * 12 + k * 4..i * 12 + k * 4 + 4].try_into().unwrap()));
                Some((0..nt).map(|t| [0, 1, 2].map(|k| vert(u32_at(8 + (t * 3 + k) * 4)))).collect())
            }
            Err(_) => None,
        };
        meshes.push((order, mesh, game_local));
    }
    meshes.sort_by_key(|m| m.0);
    let mut out = Vec::new();
    let mut goal_ly = Vec::new();
    let mut tri_box = Vec::new();  // each triangle's BVH leaf box in its body's frame: min, max
    let ray_rank = mesh_ray_ranks(meshes.iter().map(|m| &m.1));
    for (mesh_idx, (_, world, game_local)) in meshes.iter().enumerate() {
        let tr = component_translation(world);
        let local: Vec<[V; 3]> = match game_local {
            Some(l) if tr[1] != 0.0 => l.clone(),
            _ => world.iter().map(|t| t.map(|p| sub(p, tr))).collect(),
        };
        let order = bullet_leaf_order(&local);
        let quant = mesh_quant(&local);
        let local: Vec<[V; 3]> = order.iter().map(|&i| local[i]).collect();
        let leaf_start = tri_box.len();
        for t in &local {
            let (lo, hi) = leaf_box(&quant, t);
            tri_box.extend(lo.iter().chain(&hi));
        }
        let edges = internal_edge_info(&local);
        for (leaf_i, ((&i, t), (ang, flags))) in order.iter().zip(&local).zip(edges).enumerate() {
            // bits 6, 7: the mesh body's origin is (0, +-GOAL_ORIGIN_Y, 0) (goal components), else the world origin
            let origin = if tr[1] > 0.0 { 1u32 << 6 } else if tr[1] < 0.0 { 3u32 << 6 } else { 0 };
            let mut row = 0;
            if tr[1] != 0.0 {
                row = (goal_ly.len() / 3) as u32;
                goal_ly.extend(t.iter().map(|v| v[1]));
            }
            // A quantized leaf can round inside its triangle. Mark only validated non-goal boxes; goal flags use these bits for the local-y row.
            let b = &tri_box[leaf_start + leaf_i * 6..leaf_start + (leaf_i + 1) * 6];
            let leaf_contains = tr[1] == 0.0 && t.iter().all(|v| (0..3).all(|k| b[k] <= v[k] && v[k] <= b[k + 3]));
            let leaf_flag = if leaf_contains { FLAG_LEAF_CONTAINS } else { 0 };
            out.push(Tri { p: world[i], n: tri_normal(t), ang, flags: flags | origin | leaf_flag | ((mesh_idx as u32) << 8) | (ray_rank[mesh_idx] << 12) | (row << 16) });
        }
    }
    Ok((out, goal_ly, tri_box))
}

/// Grid parameters. All lengths in Bullet units.
#[derive(Clone, Copy, Debug)]
pub struct GridSpec {
    pub min: [f32; 3],
    pub cell: f32,
    pub dims: [u32; 3],
    /// Every triangle within this distance of a cell is listed in it.
    pub reach: f32,
}

impl GridSpec {
    /// Soccar: x ±4096, y ±6000 (goal backs), z 0..2048 uu.
    pub fn soccar(cell_uu: f32, reach_uu: f32) -> Self {
        let min = [-4096.0 * UU_TO_BT, -6000.0 * UU_TO_BT, 0.0];
        let max = [4096.0 * UU_TO_BT, 6000.0 * UU_TO_BT, 2048.0 * UU_TO_BT];
        let cell = cell_uu * UU_TO_BT;
        let dims = [0, 1, 2].map(|k| ((max[k] - min[k]) / cell).ceil() as u32);
        GridSpec { min, cell, dims, reach: reach_uu * UU_TO_BT }
    }
}

/// Triangle vs axis-aligned box overlap (Akenine-Möller's separating-axis test, 13 axes; offline only).
fn tri_box_overlap(centre: V, half: V, t: &[V; 3]) -> bool {
    let v = [sub(t[0], centre), sub(t[1], centre), sub(t[2], centre)];
    let e = [sub(v[1], v[0]), sub(v[2], v[1]), sub(v[0], v[2])];
    let axis_test = |a: V| -> bool {
        let p = [dot(a, v[0]), dot(a, v[1]), dot(a, v[2])];
        let r = half[0] * a[0].abs() + half[1] * a[1].abs() + half[2] * a[2].abs();
        let lo = p[0].min(p[1]).min(p[2]);
        let hi = p[0].max(p[1]).max(p[2]);
        !(lo > r || hi < -r)
    };
    for ei in e {
        for k in 0..3 {
            let mut u = [0.0f32; 3];
            u[k] = 1.0;
            let a = cross(u, ei);
            if len2(a) > 1e-12 && !axis_test(a) {
                return false;
            }
        }
    }
    for k in 0..3 {
        let lo = v[0][k].min(v[1][k]).min(v[2][k]);
        let hi = v[0][k].max(v[1][k]).max(v[2][k]);
        if lo > half[k] || hi < -half[k] {
            return false;
        }
    }
    axis_test(cross(e[0], e[1]))
}

/// For each cell, the indices of the triangles that overlap the cell box grown by `reach` on every side, sorted by
/// index. (The grown box contains every point within `reach` of the cell.)
pub fn build_cell_lists(tris: &[Tri], g: &GridSpec) -> Vec<Vec<u32>> {
    let n = (g.dims[0] * g.dims[1] * g.dims[2]) as usize;
    let mut cells = vec![Vec::new(); n];
    let half = [g.cell * 0.5 + g.reach; 3];
    for (ti, t) in tris.iter().enumerate() {
        let lo = [0, 1, 2].map(|k| t.p[0][k].min(t.p[1][k]).min(t.p[2][k]) - g.reach);
        let hi = [0, 1, 2].map(|k| t.p[0][k].max(t.p[1][k]).max(t.p[2][k]) + g.reach);
        let c0 = [0, 1, 2].map(|k| (((lo[k] - g.min[k]) / g.cell).floor().max(0.0) as u32).min(g.dims[k] - 1));
        let c1 = [0, 1, 2].map(|k| (((hi[k] - g.min[k]) / g.cell).floor().max(0.0) as u32).min(g.dims[k] - 1));
        for z in c0[2]..=c1[2] {
            for y in c0[1]..=c1[1] {
                for x in c0[0]..=c1[0] {
                    let centre = [
                        g.min[0] + (x as f32 + 0.5) * g.cell,
                        g.min[1] + (y as f32 + 0.5) * g.cell,
                        g.min[2] + (z as f32 + 0.5) * g.cell,
                    ];
                    if tri_box_overlap(centre, half, &t.p) {
                        let ci = ((z * g.dims[1] + y) * g.dims[0] + x) as usize;
                        cells[ci].push(ti as u32);
                    }
                }
            }
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(std::env::var_os("RLMAX_DATA")?).join("meshes").join("soccar");
        dir.exists().then_some(dir)
    }

    #[test]
    fn soccar_loads_and_grid_stats() {
        let Some(dir) = mesh_dir() else {
            eprintln!("RLMAX_DATA meshes not present; skipped");
            return;
        };
        let tris = load_soccar(&dir).unwrap();
        eprintln!("triangles: {}", tris.len());
        let with_edges = tris.iter().filter(|t| t.ang.iter().any(|&a| a.abs() < TAU)).count();
        eprintln!("triangles with at least one internal edge: {with_edges}");
        for (cell, reach) in [(32.0, 92.25), (64.0, 92.25), (128.0, 92.25), (64.0, 80.0), (32.0, 30.0), (64.0, 30.0), (128.0, 30.0)] {
            let g = GridSpec::soccar(cell, reach);
            let lists = build_cell_lists(&tris, &g);
            let mut counts: Vec<usize> = lists.iter().map(Vec::len).collect();
            counts.sort_unstable();
            let n = counts.len();
            let nonempty = counts.iter().filter(|&&c| c > 0).count();
            eprintln!(
                "cell {cell:>5} uu reach {reach:>6}: dims {:?} cells {n} non-empty {nonempty} p50 {} p90 {} p99 {} p99.9 {} max {}",
                g.dims, counts[n / 2], counts[n * 9 / 10], counts[n * 99 / 100], counts[n * 999 / 1000], counts[n - 1]
            );
        }
    }
}

#[cfg(test)]
mod density {
    use super::*;

    #[test]
    fn densest_cells() {
        let Some(dir) = std::env::var_os("RLMAX_DATA").map(|d| std::path::PathBuf::from(d).join("meshes/soccar")) else { return };
        let tris = load_soccar(&dir).unwrap();
        // per-mesh triangle counts and mean edge length
        let mut per_mesh = std::collections::BTreeMap::new();
        for t in &tris {
            let m = (t.flags >> 8) & 0xf;
            let e = (len2(sub(t.p[1], t.p[0])).sqrt() + len2(sub(t.p[2], t.p[1])).sqrt() + len2(sub(t.p[0], t.p[2])).sqrt()) / 3.0 * 50.0;
            let ent = per_mesh.entry(m).or_insert((0usize, 0.0f32, [f32::MAX; 3], [f32::MIN; 3]));
            ent.0 += 1;
            ent.1 += e;
            for k in 0..3 { for v in t.p { ent.2[k] = ent.2[k].min(v[k] * 50.0); ent.3[k] = ent.3[k].max(v[k] * 50.0); } }
        }
        for (m, (n, e, lo, hi)) in &per_mesh {
            eprintln!("mesh {m:2}: {n:5} tris, mean edge {:6.1} uu, bounds x {:.0}..{:.0} y {:.0}..{:.0} z {:.0}..{:.0}",
                e / *n as f32, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        }
        let g = GridSpec::soccar(64.0, 92.25);
        let lists = build_cell_lists(&tris, &g);
        let mut idx: Vec<usize> = (0..lists.len()).collect();
        idx.sort_by_key(|&i| std::cmp::Reverse(lists[i].len()));
        for &i in idx.iter().take(8) {
            let x = i as u32 % g.dims[0];
            let y = (i as u32 / g.dims[0]) % g.dims[1];
            let z = i as u32 / (g.dims[0] * g.dims[1]);
            let c = [g.min[0] + (x as f32 + 0.5) * g.cell, g.min[1] + (y as f32 + 0.5) * g.cell, g.min[2] + (z as f32 + 0.5) * g.cell];
            let meshes: std::collections::BTreeSet<u32> = lists[i].iter().map(|&t| (tris[t as usize].flags >> 8) & 0xf).collect();
            eprintln!("cell at ({:.0}, {:.0}, {:.0}) uu: {} tris, meshes {:?}", c[0] * 50.0, c[1] * 50.0, c[2] * 50.0, lists[i].len(), meshes);
        }
    }
}

/// Generator for `src/physics/sweep_tables.h` (the static half of the broadphase sweep, lead 16): each component's raw
/// shape AABB (the vertex extremes) and, with the four
/// arena planes, the static handles' quantised edges in the order `btAxisSweep3::addHandle` leaves them. Run:
/// `cargo test -p rlmax --release sweep_tables -- --ignored --nocapture`.
#[cfg(test)]
mod sweep_tables {
    use super::*;

    const HALF: f32 = f32::from_bits(0x4623_d6f6);

    fn quantize(p: f32, is_max: u16) -> u16 {
        let q = 65535.0f32 / (HALF + HALF);
        let scaled = (p - (-HALF)) * q;
        let c = if scaled >= 0.0 { scaled.min(65535.0) } else { 0.0 };
        ((c as i32 as u16) & 0xfffe) | is_max
    }

    #[test]
    #[ignore]
    fn sweep_tables() {
        let dir = std::path::PathBuf::from(std::env::var_os("RLMAX_DATA").unwrap()).join("meshes").join("soccar");
        let tris = load_soccar(&dir).unwrap();
        let mut boxes: Vec<([f32; 3], [f32; 3])> = vec![([f32::MAX; 3], [f32::MIN; 3]); 16];
        for t in &tris {
            let m = ((t.flags >> 8) & 0xf) as usize;
            for p in &t.p {
                for (k, &v) in p.iter().enumerate() {
                    boxes[m].0[k] = boxes[m].0[k].min(v);
                    boxes[m].1[k] = boxes[m].1[k].max(v);
                }
            }
        }
        let x = 4096.0 * UU_TO_BT;
        let z = 2048.0 * UU_TO_BT;
        // the game's plane bodies have a zero-thickness getAabb (binary 2a70b0), so the proxy reaches
        // only the contact breaking threshold past the surface, not a 0.2 slab
        let off = 0.02f32;
        // floor, ceiling, -x wall, +x wall (zero-thickness AABB plus the 0.02 threshold on the single axis, open behind; the dispatch gate in sweep.h keeps a 0.2 slab)
        let mut planes = [([f32::MIN; 3], [f32::MAX; 3]); 4];
        planes[0].1[2] = 0.0 + off;
        planes[1].0[2] = z - off;
        planes[2].1[0] = -x + off;
        planes[3].0[0] = x - off;
        boxes.extend(planes.iter().cloned());
        println!("// mesh raw AABBs (BT), meshes 0..15");
        for (i, b) in boxes.iter().take(16).enumerate() {
            println!("    {{{:?}f, {:?}f, {:?}f, {:?}f, {:?}f, {:?}f}},  // {i}", b.0[0], b.0[1], b.0[2], b.1[0], b.1[1], b.1[2]);
        }
        let mut edges: [Vec<(u16, u16)>; 3] = [vec![(0, 0), (0xffff, 0)], vec![(0, 0), (0xffff, 0)], vec![(0, 0), (0xffff, 0)]];
        for (h, b) in boxes.iter().enumerate() {
            for (ax, e) in edges.iter_mut().enumerate() {
                let n = e.len();
                let (qmin, qmax) = (quantize(b.0[ax], 0), quantize(b.1[ax], 1));
                e.insert(n - 1, (qmin, h as u16 + 1));
                e.insert(n, (qmax, h as u16 + 1));
                for start in [n - 1, n] {
                    let mut i = start;
                    while i > 0 && e[i].0 < e[i - 1].0 {
                        e.swap(i, i - 1);
                        i -= 1;
                    }
                }
            }
        }
        for (ax, e) in edges.iter().enumerate() {
            println!("// axis {ax}: (pos, handle)");
            let s: Vec<String> = e.iter().map(|e| format!("{{{}, {}}}", e.0, e.1)).collect();
            println!("    {{{}}},", s.join(", "));
        }
    }
}
