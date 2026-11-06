// Soccar boost pads. Frame interpolation, sweep, pad-box overlap, and respawn ordering
// are the game's; the implementation uses RLMAX state.
#pragma once

#include "car.h"
#include "consts.h"
#include "state.h"

namespace rlmax {

struct PadDef {
    float x, y, z;  // uu
    uint32_t big;
};

// the Soccar layout, sorted by y then x.
RLMAX_HD const PadDef& pad_def(int p) {
    static constexpr PadDef PADS[NUM_PADS] = {
        {0.0f, -4240.0f, 70.0f, 0},     {-1792.0f, -4184.0f, 70.0f, 0}, {1792.0f, -4184.0f, 70.0f, 0},
        {-3072.0f, -4096.0f, 73.0f, 1}, {3072.0f, -4096.0f, 73.0f, 1},  {-940.0f, -3308.0f, 70.0f, 0},
        {940.0f, -3308.0f, 70.0f, 0},   {0.0f, -2816.0f, 70.0f, 0},     {-3584.0f, -2484.0f, 70.0f, 0},
        {3584.0f, -2484.0f, 70.0f, 0},  {-1788.0f, -2302.0f, 70.0f, 0}, {1788.0f, -2302.0f, 70.0f, 0},
        {-2048.0f, -1036.0f, 70.0f, 0}, {2048.0f, -1036.0f, 70.0f, 0},  {0.0f, -1024.0f, 70.0f, 0},
        {-3584.0f, 0.0f, 73.0f, 1},     {-1024.0f, 0.0f, 70.0f, 0},     {1024.0f, 0.0f, 70.0f, 0},
        {3584.0f, 0.0f, 73.0f, 1},      {0.0f, 1024.0f, 70.0f, 0},      {-2048.0f, 1036.0f, 70.0f, 0},
        {2048.0f, 1036.0f, 70.0f, 0},   {-1788.0f, 2302.0f, 70.0f, 0},  {1788.0f, 2302.0f, 70.0f, 0},
        {-3584.0f, 2484.0f, 70.0f, 0},  {3584.0f, 2484.0f, 70.0f, 0},   {0.0f, 2816.0f, 70.0f, 0},
        {-940.0f, 3308.0f, 70.0f, 0},   {940.0f, 3308.0f, 70.0f, 0},    {-3072.0f, 4096.0f, 73.0f, 1},
        {3072.0f, 4096.0f, 73.0f, 1},   {-1792.0f, 4184.0f, 70.0f, 0},  {1792.0f, 4184.0f, 70.0f, 0},
        {0.0f, 4240.0f, 70.0f, 0},
    };
    return PADS[p];
}

constexpr uint32_t PAD_NONE = 0xFFFFFFFFu;
// the published hitbox includes a 2.5 uu skin; the pad box tests
// the published half extents. Its small pad has slightly different box and
// cylinder centers. These source constants are deliberately not rounded together.
constexpr float PAD_HITBOX_SKIN_HALF_UU = 1.25f;
constexpr float PAD_BIG_CENTER_OFFSET_UU = 1.0f;
constexpr float PAD_SMALL_BOX_CENTER_OFFSET_UU = 5.917984f;
constexpr float PAD_SMALL_CYL_CENTER_OFFSET_UU = 5.92f;
constexpr float PAD_BOX_SMALL_HALF_UU = 128.0f;
constexpr float PAD_BOX_BIG_HALF_UU = 160.0f;
constexpr float PAD_BOX_HALF_HEIGHT_UU = 64.0f;
constexpr float PAD_SWEEP_SMALL_RADIUS_UU = 176.0f;
constexpr float PAD_SWEEP_BIG_RADIUS_UU = 208.0f;
constexpr float PAD_SWEEP_HALF_HEIGHT_UU = 96.0f;
constexpr float PAD_SAT_AXIS_EPS = 1e-6f;
constexpr float PAD_SWEEP_Z_EPS_BT = 1e-7f;
constexpr float PAD_SWEEP_XY_EPS_BT2 = 1e-12f;
// Broadphase reach of one frame's sweep around a pad (uu): short sweeps stay inside it, longer ones test every pad.
constexpr float PAD_SHORT_SWEEP_UU = 240.0f;
constexpr float PAD_BROADPHASE_Z_UU = 135.0f;
// Outward-rounded z range of all pad sweep cylinders (uu): big top 168, small bottom -31.92.
constexpr float PAD_SWEEP_Z_MAX_UU = 169.0f, PAD_SWEEP_Z_MIN_UU = -32.0f;
// Coarse grid for pad_candidates: 1024 uu cells, 10 columns from x = -5, 12 rows from y = -6 (in cells).
constexpr float PAD_COARSE_CELL_UU = 1024.0f;
constexpr unsigned PAD_COARSE_X_CELLS = 10, PAD_COARSE_Y_CELLS = 12;
constexpr int PAD_COARSE_X_ORIGIN = 5, PAD_COARSE_Y_ORIGIN = 6;
// Fine grid for pad_cell_slots: 512 uu cells over the field, 16 columns by 20 rows.
constexpr float PAD_CELL_UU = 512.0f;
constexpr uint32_t PAD_CELL_COLUMNS = 16u, PAD_CELL_ROWS = 20u;
// Pickup cooldowns in ticks, and the pad-pick record: a pad index and a car index packed in six-bit fields.
constexpr uint32_t PAD_COOLDOWN_BIG_TICKS = 1200u, PAD_COOLDOWN_SMALL_TICKS = 480u;
constexpr uint32_t PAD_FIELD_BITS = 6u, PAD_FIELD_MASK = 63u;
// The game quantises a car's position to 1/100 uu; a pickup fraction is stored as sixteen bits.
constexpr float PAD_POS_QUANT = 100.0f;
constexpr float PAD_FRAC_Q_MAX = 65535.0f;
constexpr uint32_t PAD_FRAC_XOR = 0x3f000000u;  // the bits of 0.5f: the stored fractions order as unsigned integers

// Oriented car box against the frame-end pad box. The 15-axis separating
// test and geometry are the game's.
RLMAX_HD bool pad_box_overlap(const PadDef& pad, V3 pos_uu, const M3& rotation, const CarGeo& geo) {
    const float radius = pad.big ? PAD_BOX_BIG_HALF_UU : PAD_BOX_SMALL_HALF_UU;
    const float pad_half[3] = {radius, radius, PAD_BOX_HALF_HEIGHT_UU};
    const V3 car_half = (geo.hitbox_half + v3s(geo.margin)) * c::BT_TO_UU - v3s(PAD_HITBOX_SKIN_HALF_UU);
    const float car_extents[3] = {car_half.x, car_half.y, car_half.z};
    const V3 centre = pos_uu + rotation * (geo.hitbox_off * c::BT_TO_UU);
    const float delta[3] = {centre.x - pad.x, centre.y - pad.y,
                            centre.z - (pad.z - (pad.big ? PAD_BIG_CENTER_OFFSET_UU : PAD_SMALL_BOX_CENTER_OFFSET_UU))};
    const V3 axes[3] = {rotation.c0, rotation.c1, rotation.c2};
    float axis[3][3], abs_axis[3][3];
    for (int j = 0; j < 3; ++j) {
        const float components[3] = {axes[j].x, axes[j].y, axes[j].z};
        for (int i = 0; i < 3; ++i) { axis[i][j] = components[i]; abs_axis[i][j] = absf(components[i]) + PAD_SAT_AXIS_EPS; }
    }
    for (int i = 0; i < 3; ++i) {
        if (absf(delta[i]) > pad_half[i] + car_extents[0] * abs_axis[i][0] + car_extents[1] * abs_axis[i][1] + car_extents[2] * abs_axis[i][2]) return false;
    }
    for (int j = 0; j < 3; ++j) {
        if (absf(delta[0] * axis[0][j] + delta[1] * axis[1][j] + delta[2] * axis[2][j]) >
            pad_half[0] * abs_axis[0][j] + pad_half[1] * abs_axis[1][j] + pad_half[2] * abs_axis[2][j] + car_extents[j]) return false;
    }
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        const int i1 = (i + 1) % 3, i2 = (i + 2) % 3, j1 = (j + 1) % 3, j2 = (j + 2) % 3;
        const float pad_radius = pad_half[i1] * abs_axis[i2][j] + pad_half[i2] * abs_axis[i1][j];
        const float car_radius = car_extents[j1] * abs_axis[i][j2] + car_extents[j2] * abs_axis[i][j1];
        if (absf(delta[i2] * axis[i1][j] - delta[i1] * axis[i2][j]) > pad_radius + car_radius) return false;
    }
    return true;
}

// Frame sweep and respawn ordering as the game's. Coordinates are in BT.
RLMAX_HD bool pad_sweep(const PadDef& pad, V3 from_bt, V3 to_bt, float& enter) {
    const float radius = (pad.big ? PAD_SWEEP_BIG_RADIUS_UU : PAD_SWEEP_SMALL_RADIUS_UU) * c::UU_TO_BT;
    const float z0 = from_bt.z - (pad.z - (pad.big ? PAD_BIG_CENTER_OFFSET_UU : PAD_SMALL_CYL_CENTER_OFFSET_UU)) * c::UU_TO_BT;
    const float dz = to_bt.z - from_bt.z;
    float lo = 0.0f, hi = 1.0f;
    if (absf(dz) < PAD_SWEEP_Z_EPS_BT) { if (absf(z0) >= PAD_SWEEP_HALF_HEIGHT_UU * c::UU_TO_BT) return false; }
    else {
        const float lower = (-PAD_SWEEP_HALF_HEIGHT_UU * c::UU_TO_BT - z0) / dz;
        const float upper = (PAD_SWEEP_HALF_HEIGHT_UU * c::UU_TO_BT - z0) / dz;
        lo = fmax_(lo, fmin_(lower, upper)); hi = fmin_(hi, fmax_(lower, upper));
    }
    const float x = from_bt.x - pad.x * c::UU_TO_BT, y = from_bt.y - pad.y * c::UU_TO_BT;
    const float dx = to_bt.x - from_bt.x, dy = to_bt.y - from_bt.y;
    const float movement2 = dx * dx + dy * dy, projected = x * dx + y * dy, distance = x * x + y * y - radius * radius;
    if (movement2 < PAD_SWEEP_XY_EPS_BT2) { enter = lo; return distance < 0.0f && lo < hi; }
    const float disc = projected * projected - movement2 * distance;
    if (disc <= 0.0f) return false;
    const float root = sqrtf(disc);
    enter = fmax_(lo, (-projected - root) / movement2);
    return enter < fmin_(hi, (-projected + root) / movement2);
}
RLMAX_HD V3 pad_quant_pos(V3 bt) {
    V3 u = bt * c::BT_TO_UU;
    return v3(q_pos(u.x, PAD_POS_QUANT), q_pos(u.y, PAD_POS_QUANT), q_pos(u.z, PAD_POS_QUANT));
}
RLMAX_HD bool pad_can_receive(const CarState& cs) {
    return !(cs.flags & CF_IS_DEMOED) && !(cs.ctl.buttons & BTN_UNDRIVEN) && cs.boost < c::BOOST_MAX;
}
RLMAX_HD int pad_first_set(uint64_t bits) {
#if defined(__CUDA_ARCH__)
    return __ffsll((long long)bits) - 1;
#elif defined(__clang__) || defined(__GNUC__)
    return __builtin_ctzll(bits);
#else
    int n = 0; while (!(bits & 1u)) { bits >>= 1; ++n; } return n;
#endif
}
// 1024 uu grid: a short sweep can reach only the frame-end cell and its neighbors.
// Long sweeps fall back to all pads. Masks are the Soccar layout in pad_def order.
RLMAX_HD uint64_t pad_candidates(V3 end_uu, V3 delta_bt) {
    if (len2(delta_bt) > (PAD_SHORT_SWEEP_UU * c::UU_TO_BT) * (PAD_SHORT_SWEEP_UU * c::UU_TO_BT)) return (uint64_t(1) << NUM_PADS) - 1;
    static constexpr uint64_t x_column_pads[PAD_COARSE_X_CELLS] = {
        0x000000000ULL, 0x021008108ULL, 0x021109108ULL, 0x0a151950aULL, 0x29c5954e3ULL,
        0x39ccb4ce7ULL, 0x31caa68e5ULL, 0x142a62a14ULL, 0x042242210ULL, 0x042040210ULL
    };
    static constexpr uint64_t y_row_pads[PAD_COARSE_Y_CELLS] = {
        0x000000000ULL, 0x00000001fULL, 0x00000007fULL, 0x000000fffULL, 0x000007fe0ULL, 0x00007ff80ULL,
        0x0003ff000ULL, 0x007ff8000ULL, 0x01ff80000ULL, 0x3ffc00000ULL, 0x3f8000000ULL, 0x3e0000000ULL
    };
    const unsigned x = (unsigned)((int)(end_uu.x / PAD_COARSE_CELL_UU) + PAD_COARSE_X_ORIGIN);
    const unsigned y = (unsigned)((int)(end_uu.y / PAD_COARSE_CELL_UU) + PAD_COARSE_Y_ORIGIN);
    return x < PAD_COARSE_X_CELLS && y < PAD_COARSE_Y_CELLS ? x_column_pads[x] & y_row_pads[y] : 0;
}
// A 512 uu cell and a <=240 uu sweep can reach only pads within 480 uu
// of that cell's square (the extra reach is the conservative 240 uu broadphase).
// Soccar has at most two such pads per cell. Each six-bit slot is pad index + 1;
// zero means empty. The table and cached cell/slots use no additional state bytes.
// The first car also holds an expiry-check deadline in unused bits. Each
// deadline is at most 255 ticks away, earlier than any new 480-tick cooldown.
constexpr uint32_t PAD_EXPIRY_VALID = 0x40000000u;
constexpr uint32_t PAD_EXPIRY_SHIFT = 22u;
constexpr uint32_t PAD_EXPIRY_TICK_MASK = 255u;
constexpr uint32_t PAD_EXPIRY_META = PAD_EXPIRY_VALID | (PAD_EXPIRY_TICK_MASK << PAD_EXPIRY_SHIFT);
// A demolished or teleported car has no pad overlap carried from its last frame. Slot 0 also holds the arena's expiry cache.
RLMAX_HD void pad_frame_forget(CarState& cs, int slot) {
    cs.pad_frame_valid = slot == 0 ? cs.pad_frame_valid & PAD_EXPIRY_META : 0;
    cs.pad_touch = PAD_NONE;
}
constexpr uint32_t PAD_CELL_CACHE_TAG = 0x80000000u;
constexpr uint32_t PAD_CELL_MASK = 0x1ffu;
constexpr uint32_t PAD_SLOT_MASK = 0xfffu;
RLMAX_HD uint16_t pad_cell_slots(uint32_t cell) {
    static constexpr uint16_t SLOTS[320] = {
        0x000, 0x000, 0x000, 0x002, 0x002, 0x002, 0x000, 0x001, 0x001, 0x000, 0x003, 0x003, 0x003, 0x000, 0x000, 0x000,
        0x000, 0x004, 0x004, 0x002, 0x002, 0x002, 0x000, 0x001, 0x001, 0x000, 0x003, 0x003, 0x003, 0x005, 0x005, 0x000,
        0x000, 0x004, 0x004, 0x002, 0x002, 0x182, 0x006, 0x181, 0x1c1, 0x007, 0x1c3, 0x003, 0x003, 0x005, 0x005, 0x000,
        0x000, 0x000, 0x000, 0x000, 0x000, 0x006, 0x006, 0x206, 0x207, 0x007, 0x007, 0x000, 0x000, 0x000, 0x000, 0x000,
        0x009, 0x009, 0x000, 0x00b, 0x00b, 0x2c6, 0x006, 0x206, 0x207, 0x007, 0x307, 0x00c, 0x00c, 0x000, 0x00a, 0x00a,
        0x009, 0x009, 0x000, 0x00b, 0x00b, 0x00b, 0x000, 0x008, 0x008, 0x000, 0x00c, 0x00c, 0x00c, 0x000, 0x00a, 0x00a,
        0x009, 0x009, 0x000, 0x00b, 0x00b, 0x00b, 0x000, 0x000, 0x000, 0x000, 0x00c, 0x00c, 0x00c, 0x000, 0x00a, 0x00a,
        0x000, 0x000, 0x000, 0x00d, 0x00d, 0x000, 0x000, 0x00f, 0x00f, 0x000, 0x000, 0x00e, 0x00e, 0x000, 0x000, 0x000,
        0x000, 0x000, 0x000, 0x00d, 0x00d, 0x000, 0x000, 0x00f, 0x00f, 0x000, 0x000, 0x00e, 0x00e, 0x000, 0x000, 0x000,
        0x010, 0x010, 0x000, 0x000, 0x000, 0x011, 0x011, 0x000, 0x000, 0x012, 0x012, 0x000, 0x000, 0x000, 0x013, 0x013,
        0x010, 0x010, 0x000, 0x000, 0x000, 0x011, 0x011, 0x000, 0x000, 0x012, 0x012, 0x000, 0x000, 0x000, 0x013, 0x013,
        0x000, 0x000, 0x000, 0x015, 0x015, 0x000, 0x000, 0x014, 0x014, 0x000, 0x000, 0x016, 0x016, 0x000, 0x000, 0x000,
        0x000, 0x000, 0x000, 0x015, 0x015, 0x000, 0x000, 0x014, 0x014, 0x000, 0x000, 0x016, 0x016, 0x000, 0x000, 0x000,
        0x019, 0x019, 0x000, 0x017, 0x017, 0x017, 0x000, 0x000, 0x000, 0x000, 0x018, 0x018, 0x018, 0x000, 0x01a, 0x01a,
        0x019, 0x019, 0x000, 0x017, 0x017, 0x017, 0x000, 0x01b, 0x01b, 0x000, 0x018, 0x018, 0x018, 0x000, 0x01a, 0x01a,
        0x019, 0x019, 0x000, 0x017, 0x017, 0x717, 0x01c, 0x71b, 0x75b, 0x01d, 0x758, 0x018, 0x018, 0x000, 0x01a, 0x01a,
        0x000, 0x000, 0x000, 0x000, 0x000, 0x01c, 0x01c, 0x71b, 0x75b, 0x01d, 0x01d, 0x000, 0x000, 0x000, 0x000, 0x000,
        0x000, 0x01e, 0x01e, 0x020, 0x020, 0x81c, 0x01c, 0x89c, 0x89d, 0x01d, 0x85d, 0x021, 0x021, 0x01f, 0x01f, 0x000,
        0x000, 0x01e, 0x01e, 0x020, 0x020, 0x020, 0x000, 0x022, 0x022, 0x000, 0x021, 0x021, 0x021, 0x01f, 0x01f, 0x000,
        0x000, 0x000, 0x000, 0x020, 0x020, 0x020, 0x000, 0x022, 0x022, 0x000, 0x021, 0x021, 0x021, 0x000, 0x000, 0x000,
    };
    return SLOTS[cell];
}
// Cheap bounds precede the cylinder intersection; inputs from/pos are uu and
// from_bt/to_bt are the exact frame endpoints used by the sweep.
RLMAX_HD void pad_accumulate_sweep(int p, V3 from, V3 pos, V3 from_bt, V3 to_bt, uint64_t& hits) {
    const PadDef& pad = pad_def(p);
    if (pad.x < fmin_(from.x, pos.x) - PAD_SHORT_SWEEP_UU || pad.x > fmax_(from.x, pos.x) + PAD_SHORT_SWEEP_UU ||
        pad.y < fmin_(from.y, pos.y) - PAD_SHORT_SWEEP_UU || pad.y > fmax_(from.y, pos.y) + PAD_SHORT_SWEEP_UU ||
        pad.z < fmin_(from.z, pos.z) - PAD_BROADPHASE_Z_UU || pad.z > fmax_(from.z, pos.z) + PAD_BROADPHASE_Z_UU) return;
    float enter;
    if (pad_sweep(pad, from_bt, to_bt, enter)) hits |= uint64_t(1) << p;
}
RLMAX_HD uint32_t pad_encode_frac(float f) { union { float f; uint32_t u; } x; x.f = f; return x.u ^ PAD_FRAC_XOR; }
RLMAX_HD float pad_decode_frac(uint32_t bits) { union { float f; uint32_t u; } x; x.u = bits ^ PAD_FRAC_XOR; return x.f; }
RLMAX_HD uint16_t pad_frac_q(float f) { return (uint16_t)(fmin_(fmax_(f, 0.0f), 1.0f) * PAD_FRAC_Q_MAX + 0.5f); }

template <int NC>
RLMAX_HD void pads_update(ArenaState<NC>& s, const Body* bodies, const CarGeo* geo, const V3* tick_start, int32_t* picked = nullptr) {
    const float frac = pad_decode_frac(s.pad_frame_bits);
    if (frac < 0.0f) {
        // A deadline reached on an earlier tick must be rechecked at the next frame end.
        uint32_t& meta = s.cars[0].pad_frame_valid;
        if ((meta & PAD_EXPIRY_VALID) && ((meta >> PAD_EXPIRY_SHIFT) & PAD_EXPIRY_TICK_MASK) == (s.tick & PAD_EXPIRY_TICK_MASK)) meta &= ~PAD_EXPIRY_VALID;
        return;
    }
    uint64_t locked = 0;
    uint16_t pending[2*NC]; int pending_count = 0;
    for (int k = 0; k < NC; ++k) {
        const int i = visit_slot<NC>(s.visit, k);
        CarState& cs = s.cars[i];
        if (cs.flags & CF_IS_DEMOED) {
            pad_frame_forget(cs, i); continue;
        }
        const V3 start = pad_quant_pos(tick_start[1+i]);
        const V3 end = pad_quant_pos(cs.pos);
        const V3 pos = start + (end - start) * frac;
        const V3 to_bt = pos * c::UU_TO_BT;
        const uint32_t old_frame_cache = cs.pad_frame_valid;
        const V3 from_bt = (old_frame_cache & 1u) ? cs.pad_frame_pos : to_bt - (end - start) * c::UU_TO_BT;
        const V3 from = from_bt * c::BT_TO_UU;
        cs.pad_frame_pos = to_bt;
        if (!(old_frame_cache & 1u)) cs.pad_frame_valid = (old_frame_cache & PAD_EXPIRY_META) | 1u;
        uint64_t hits = 0;
        if (fmin_(from.z,pos.z) <= PAD_SWEEP_Z_MAX_UU && fmax_(from.z,pos.z) >= PAD_SWEEP_Z_MIN_UU) {
            const V3 movement = to_bt - from_bt;
            const float short_limit = PAD_SHORT_SWEEP_UU * c::UU_TO_BT;
            const bool short_sweep = len2(movement) <= short_limit * short_limit;
            const bool inside_grid = pos.x >= -c::ARENA_X_UU && pos.x < c::ARENA_X_UU && pos.y >= -c::ARENA_Y_UU && pos.y < c::ARENA_Y_UU;
            if (short_sweep && inside_grid) {
                const uint32_t cell_x = (uint32_t)((pos.x + c::ARENA_X_UU) * (1.0f / PAD_CELL_UU));
                const uint32_t cell_y = (uint32_t)((pos.y + c::ARENA_Y_UU) * (1.0f / PAD_CELL_UU));
                // Clamp the rounded upper boundary into the last cell; the table's 480 uu reach is conservative.
                const uint32_t cell = (cell_y < PAD_CELL_ROWS ? cell_y : PAD_CELL_ROWS - 1) * PAD_CELL_COLUMNS +
                                      (cell_x < PAD_CELL_COLUMNS ? cell_x : PAD_CELL_COLUMNS - 1);
                uint16_t slots;
                if ((old_frame_cache & PAD_CELL_CACHE_TAG) && ((old_frame_cache >> 1) & PAD_CELL_MASK) == cell) {
                    slots = (uint16_t)((old_frame_cache >> 10) & PAD_SLOT_MASK);
                } else {
                    slots = pad_cell_slots(cell);
                    cs.pad_frame_valid = (old_frame_cache & PAD_EXPIRY_META) | PAD_CELL_CACHE_TAG | (uint32_t(slots) << 10) | (cell << 1) | 1u;
                }
                const uint32_t first = slots & PAD_FIELD_MASK, second = slots >> PAD_FIELD_BITS;
                // Inactive pads cannot lock during the contact pass. Respawn remains a separate pass below.
                if (cs.pad_touch == PAD_NONE &&
                    (!first || s.pad_gave_tick[first - 1] != PAD_NONE || ((locked >> (first - 1)) & 1u)) &&
                    (!second || s.pad_gave_tick[second - 1] != PAD_NONE || ((locked >> (second - 1)) & 1u))) continue;
                if (first) pad_accumulate_sweep((int)first - 1, from, pos, from_bt, to_bt, hits);
                if (second) pad_accumulate_sweep((int)second - 1, from, pos, from_bt, to_bt, hits);
            } else {
                // Long sweeps and positions outside the cached field use the existing conservative grid.
                for (uint64_t m = pad_candidates(pos, movement); m; m &= m - 1)
                    pad_accumulate_sweep(pad_first_set(m), from, pos, from_bt, to_bt, hits);
            }
        }
        if (!hits && cs.pad_touch == PAD_NONE) continue;
        const V3 spin = cs.ang_vel * (-(1.0f - frac) * c::DT);
        M3 r = bodies[1+i].r;
        r.c0 += cross(spin, r.c0); r.c1 += cross(spin, r.c1); r.c2 += cross(spin, r.c2);
        if (cs.pad_touch != PAD_NONE && !((hits >> cs.pad_touch) & 1u) &&
            s.pad_gave_tick[cs.pad_touch] == PAD_NONE && !((locked >> cs.pad_touch) & 1u) &&
            !pad_box_overlap(pad_def((int)cs.pad_touch), pos, r, geo[i])) cs.pad_touch = PAD_NONE;
        const bool multi = hits && (hits & (hits - 1));
        while (hits) {
            int hit = pad_first_set(hits);
            if (multi) {
                float best = 2.0f;
                for (uint64_t m = hits; m; m &= m - 1) {
                    const int p = pad_first_set(m); float t;
                    pad_sweep(pad_def(p), from_bt, to_bt, t);
                    if (t < best) { hit = p; best = t; }
                }
            }
            hits &= ~(uint64_t(1) << hit);
            if (s.pad_gave_tick[hit] != PAD_NONE || ((locked >> hit) & 1u)) continue;
            if (cs.pad_touch == (uint32_t)hit) {
                if (!pad_box_overlap(pad_def(hit), pos, r, geo[i])) cs.pad_touch = PAD_NONE;
                continue;
            }
            if (pad_can_receive(cs)) {
                cs.pad_touch = (uint32_t)hit; locked |= uint64_t(1) << hit;
                if (!multi) pending[pending_count++] = (uint16_t)(hit | (i << PAD_FIELD_BITS));
                if (multi) {
                    const PadDef& d = pad_def(hit);
                    cs.boost = fmin_(cs.boost + (d.big ? c::PAD_AMOUNT_BIG : c::PAD_AMOUNT_SMALL), c::BOOST_MAX);
                    s.pad_gave_tick[hit] = s.tick; s.pad_pickup_frac[hit] = pad_frac_q(frac);
                    if (picked) picked[i] = hit;
                    for (int j = 0; j < NC; ++j) if (s.cars[j].pad_touch == (uint32_t)hit) s.cars[j].pad_touch = PAD_NONE;
                }
            } else if (pad_box_overlap(pad_def(hit), pos, r, geo[i])) cs.pad_touch = (uint32_t)hit;
        }
    }
    // No pad can expire before the saved deadline. A newly collected pad has
    // at least 480 ticks of cooldown, so it cannot precede a <=255-tick deadline.
    uint32_t& expiry_meta = s.cars[0].pad_frame_valid;
    const bool expiry_due = !(expiry_meta & PAD_EXPIRY_VALID) ||
                            ((expiry_meta >> PAD_EXPIRY_SHIFT) & PAD_EXPIRY_TICK_MASK) == (s.tick & PAD_EXPIRY_TICK_MASK);
    if (expiry_due) {
        uint32_t next_check = PAD_EXPIRY_TICK_MASK;
        for (int p = 0; p < NUM_PADS; ++p) {
            if (s.pad_gave_tick[p] == PAD_NONE) continue;
            const uint32_t age = s.tick - s.pad_gave_tick[p], period = pad_def(p).big ? PAD_COOLDOWN_BIG_TICKS : PAD_COOLDOWN_SMALL_TICKS;
            if (age < period) {
                const uint32_t remaining = period - age;
                if (remaining < next_check) next_check = remaining;
                continue;
            }
            if (age == period && s.pad_pickup_frac[p] >= pad_frac_q(frac)) {
                next_check = 1; // the next frame's fraction may break the tie
                continue;
            }
            s.pad_gave_tick[p] = PAD_NONE;
            for (int k = NC - 1; k >= 0; --k) {
                const int i = visit_slot<NC>(s.visit, k);
                if (!(s.cars[i].flags & CF_IS_DEMOED)) {
                    const V3 start = pad_quant_pos(tick_start[1+i]), end = pad_quant_pos(s.cars[i].pos);
                    const V3 pos = start + (end-start)*frac;
                    const V3 spin = s.cars[i].ang_vel * (-(1.0f-frac)*c::DT);
                    M3 r = bodies[1+i].r;
                    r.c0 += cross(spin,r.c0); r.c1 += cross(spin,r.c1); r.c2 += cross(spin,r.c2);
                    if (pad_box_overlap(pad_def(p), pos, r, geo[i])) s.cars[i].pad_touch = p;
                }
            }
        }
        expiry_meta = (expiry_meta & ~PAD_EXPIRY_META) | PAD_EXPIRY_VALID |
                      (((s.tick + next_check) & PAD_EXPIRY_TICK_MASK) << PAD_EXPIRY_SHIFT);
    }
    for (int k = NC - 1; k >= 0; --k) {
        const int i = visit_slot<NC>(s.visit, k); CarState& cs = s.cars[i];
        if (cs.pad_touch != PAD_NONE && s.pad_gave_tick[cs.pad_touch] == PAD_NONE && !((locked >> cs.pad_touch) & 1u) && pad_can_receive(cs)) { locked |= uint64_t(1) << cs.pad_touch; pending[pending_count++] = (uint16_t)(cs.pad_touch | (i << PAD_FIELD_BITS)); }
    }
    // Visit only locked pads, in the same ascending pad order as the full scan.
    for (uint64_t remaining = locked; remaining; remaining &= remaining - 1) {
        const int p = pad_first_set(remaining);
        if (s.pad_gave_tick[p] != PAD_NONE) continue;
        int i = -1;
        for (int j=0;j<pending_count;++j) if ((pending[j] & PAD_FIELD_MASK) == p) { i = pending[j] >> PAD_FIELD_BITS; break; }
        if (i < 0) continue;
        CarState& cs = s.cars[i]; const PadDef& d = pad_def(p);
        cs.boost = fmin_(cs.boost + (d.big ? c::PAD_AMOUNT_BIG : c::PAD_AMOUNT_SMALL), c::BOOST_MAX);
        s.pad_gave_tick[p] = s.tick; s.pad_pickup_frac[p] = pad_frac_q(frac);
        if (picked) picked[i] = p;
        for (int j = 0; j < NC; ++j) if (s.cars[j].pad_touch == (uint32_t)p) s.cars[j].pad_touch = PAD_NONE;
    }
}

}  // namespace rlmax
