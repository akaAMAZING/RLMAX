// RLMAX step kernel: one thread per arena, `ticks` ticks per launch, the same tick as the CPU build (src/physics/tick.h).
// Built under exact math (no FMA contraction, IEEE division and square root) as the oracle build: its state must equal
// the CPU build's on every word (DESIGN §11, P2 gate).

#include "physics/tick.h"
#include "physics/extract.h"

using namespace rlmax;

#if defined(RLMAX_PROFILE)
// the profiling build's stage cycles (tick.h RLMAX_PROF), read and cleared by the host
extern "C" __global__ void rlmax_prof_read(unsigned long long* out) {
    for (int k = 0; k < 32; k++) {
        out[k] = g_prof[k];
        g_prof[k] = 0;
    }
}
#endif

// EPA's working storage is thread-local: local memory is interleaved per lane by the hardware, which measured faster
// than a per-arena global buffer (EPA runs on most resting contacts through the pair detector's 0.01 catch). It is
// the one whitelisted local-memory user in the SASS scan. Nothing recurses, so the frame is static and the driver sizes
// each thread's local memory from it.

// Register budget of the kernels that run the tick, per arch and car count. sm_86 (the 3080) and sm_80 (A100, A30, the
// same register file): 224 / 255 / 255, the fastest measured on the 3080 at 8,192 and within noise of 184 / 224 / 224
// and 152 / 224 / 224 (5 Nov, two seeds, 1v1 8.22, 7.49 / 8.09, 6.95 / 8.02, 7.91); the tick no longer fits without spills
// at any of them. Other archs: ptxas's own choice under the 128-thread launch bound until that card is measured.
// RLMAX_REGS_<NC> (build.rs RLMAX_REGS) overrides the budget for measurement builds. nvcc takes __maxnreg__ or
// __launch_bounds__, not both; the block is 128 threads either way.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 800)
#define RLMAX_SM86_REGS_2 224
#define RLMAX_SM86_REGS_3 255
#define RLMAX_SM86_REGS_4 255
#define RLMAX_SM86_REGS_5 255
#define RLMAX_SM86_REGS_6 255
#define RLMAX_SM86_REGS_8 255
#endif
#if defined(RLMAX_REGS_2)
#define RLMAX_TICK_BOUNDS_2 __maxnreg__(RLMAX_REGS_2)
#elif defined(RLMAX_SM86_REGS_2)
#define RLMAX_TICK_BOUNDS_2 __maxnreg__(RLMAX_SM86_REGS_2)
#else
#define RLMAX_TICK_BOUNDS_2 __launch_bounds__(128)
#endif
#if defined(RLMAX_REGS_3)
#define RLMAX_TICK_BOUNDS_3 __maxnreg__(RLMAX_REGS_3)
#elif defined(RLMAX_SM86_REGS_3)
#define RLMAX_TICK_BOUNDS_3 __maxnreg__(RLMAX_SM86_REGS_3)
#else
#define RLMAX_TICK_BOUNDS_3 __launch_bounds__(128)
#endif
#if defined(RLMAX_REGS_4)
#define RLMAX_TICK_BOUNDS_4 __maxnreg__(RLMAX_REGS_4)
#elif defined(RLMAX_SM86_REGS_4)
#define RLMAX_TICK_BOUNDS_4 __maxnreg__(RLMAX_SM86_REGS_4)
#else
#define RLMAX_TICK_BOUNDS_4 __launch_bounds__(128)
#endif
#if defined(RLMAX_REGS_5)
#define RLMAX_TICK_BOUNDS_5 __maxnreg__(RLMAX_REGS_5)
#elif defined(RLMAX_SM86_REGS_5)
#define RLMAX_TICK_BOUNDS_5 __maxnreg__(RLMAX_SM86_REGS_5)
#else
#define RLMAX_TICK_BOUNDS_5 __launch_bounds__(128)
#endif
#if defined(RLMAX_REGS_6)
#define RLMAX_TICK_BOUNDS_6 __maxnreg__(RLMAX_REGS_6)
#elif defined(RLMAX_SM86_REGS_6)
#define RLMAX_TICK_BOUNDS_6 __maxnreg__(RLMAX_SM86_REGS_6)
#else
#define RLMAX_TICK_BOUNDS_6 __launch_bounds__(128)
#endif

#if defined(RLMAX_REGS_8)
#define RLMAX_TICK_BOUNDS_8 __maxnreg__(RLMAX_REGS_8)
#elif defined(RLMAX_SM86_REGS_8)
#define RLMAX_TICK_BOUNDS_8 __maxnreg__(RLMAX_SM86_REGS_8)
#else
#define RLMAX_TICK_BOUNDS_8 __launch_bounds__(128)
#endif

template <int NC, bool HasPsyclops = false>
__device__ __forceinline__ void step_arenas(uint32_t* states, uint32_t n, uint32_t ticks, const ArenaData& ad) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ArenaState<NC>* a = reinterpret_cast<ArenaState<NC>*>(states) + i;
    EpaScratch scratch;
    GeoCache<NC> gc;  // the cars' geometry derived once per launch, not per tick (tick.h)
    gc.reset();
    for (uint32_t t = 0; t < ticks; t++) tick<NC, HasPsyclops>(*a, ad, scratch, gc);
}

extern "C" __global__ void RLMAX_TICK_BOUNDS_2 rlmax_step_2(uint32_t* states, uint32_t n, uint32_t ticks, ArenaData ad) {
    step_arenas<2>(states, n, ticks, ad);
}
extern "C" __global__ void RLMAX_TICK_BOUNDS_3 rlmax_step_3(uint32_t* states, uint32_t n, uint32_t ticks, ArenaData ad) {
    step_arenas<3>(states, n, ticks, ad);
}
extern "C" __global__ void RLMAX_TICK_BOUNDS_4 rlmax_step_4(uint32_t* states, uint32_t n, uint32_t ticks, ArenaData ad) {
    step_arenas<4>(states, n, ticks, ad);
}
extern "C" __global__ void RLMAX_TICK_BOUNDS_5 rlmax_step_5(uint32_t* states, uint32_t n, uint32_t ticks, ArenaData ad) {
    step_arenas<5>(states, n, ticks, ad);
}
extern "C" __global__ void RLMAX_TICK_BOUNDS_6 rlmax_step_6(uint32_t* states, uint32_t n, uint32_t ticks, ArenaData ad) {
    step_arenas<6>(states, n, ticks, ad);
}

extern "C" __global__ void RLMAX_TICK_BOUNDS_8 rlmax_step_8(uint32_t* states, uint32_t n, uint32_t ticks, ArenaData ad) {
    step_arenas<8>(states, n, ticks, ad);
}

// One decision for every arena, the trainer's step: an arena whose last decision scored is reset to a kickoff first
// (the goal reset at the decision boundary, counted in `goals`), then each car's action row (one byte per car) is
// looked up in the action table (8 floats a row: throttle, steer, pitch, yaw, roll, jump, boost, handbrake) and `ticks`
// ticks run. Without rows (null) the table holds one row per car, arena-major: raw controls, for action spaces past
// 256 rows (evaluation of other bots); the same floats give the same physics.
template <int NC, bool HasPsyclops = false>
__device__ __forceinline__ void decide_arenas(uint32_t* states, uint32_t n, uint32_t ticks, const ArenaData& ad,
                                              const uint8_t* rows, const float* table, uint32_t* goals) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ArenaState<NC>* a = reinterpret_cast<ArenaState<NC>*>(states) + i;
    if (a->events & 3u) {
        kickoff_reset<NC>(*a);
        goals[i] += 1;
    }
    for (int c = 0; c < NC; c++) {
        const float* t = table + 8 * (rows ? (uint32_t)rows[i * NC + c] : i * NC + c);
        Controls& ctl = a->cars[c].ctl;
        ctl.throttle = t[0];
        ctl.steer = t[1];
        ctl.pitch = t[2];
        ctl.yaw = t[3];
        ctl.roll = t[4];
        ctl.buttons = (t[5] != 0.0f ? BTN_JUMP : 0u) | (t[6] != 0.0f ? BTN_BOOST : 0u) | (t[7] != 0.0f ? BTN_HANDBRAKE : 0u);
    }
    EpaScratch scratch;
    GeoCache<NC> gc;  // the cars' geometry derived once per launch, not per tick (tick.h)
    gc.reset();
    for (uint32_t t = 0; t < ticks; t++) tick<NC, HasPsyclops>(*a, ad, scratch, gc);
}

// One decision of a short rollout (RLOO, search): as decide, but a goal stops the arena where it is, its final state
// kept readable. stop[i]: bits 0..15 the tick of this call on which the goal fell (1-based), 0xFFFF if none, 0 if the
// arena had already stopped before the call; bits 16..17 the side that scored (events: 1 blue, 2 orange). A stopped
// arena does not step and ignores its row.
template <int NC, bool HasPsyclops = false>
__device__ __forceinline__ void rollout_arenas(uint32_t* states, uint32_t n, uint32_t ticks, const ArenaData& ad,
                                               const uint8_t* rows, const float* table, uint32_t* stop) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ArenaState<NC>* a = reinterpret_cast<ArenaState<NC>*>(states) + i;
    if (a->events & 3u) {
        stop[i] = (a->events & 3u) << 16;
        return;
    }
    for (int c = 0; c < NC; c++) {
        const float* t = table + 8 * (uint32_t)rows[i * NC + c];
        Controls& ctl = a->cars[c].ctl;
        ctl.throttle = t[0];
        ctl.steer = t[1];
        ctl.pitch = t[2];
        ctl.yaw = t[3];
        ctl.roll = t[4];
        ctl.buttons = (t[5] != 0.0f ? BTN_JUMP : 0u) | (t[6] != 0.0f ? BTN_BOOST : 0u) | (t[7] != 0.0f ? BTN_HANDBRAKE : 0u);
    }
    EpaScratch scratch;
    GeoCache<NC> gc;
    gc.reset();
    uint32_t out = 0xFFFFu;
    for (uint32_t t = 0; t < ticks; t++) {
        tick<NC, HasPsyclops>(*a, ad, scratch, gc);
        if (a->events & 3u) {
            out = (t + 1u) | ((a->events & 3u) << 16);
            break;
        }
    }
    stop[i] = out;
}

// The trainer's view: one row of raw physical values per arena (physics/extract.h).
template <int NC>
__device__ __forceinline__ void extract_arenas(const uint32_t* states, uint32_t n, float* out) {
    // a block is one warp and 32 arenas: each lane builds its arena's row in shared memory (the stride padded to an odd
    // width, so the lanes' writes fall in different banks), then the warp stores the 32 rows, contiguous in `out`, as
    // whole 128-byte lines; a lane writing its own row directly stored 4 bytes per 32-byte sector (65,536 arenas: 1.9 ms
    // for 1v1, against 60 us at 8,192 while the output still fit in L2)
    constexpr int R = extract_row_floats<NC>(), RP = R | 1;
    // eight cars' rows would pass the card's 48 KB of shared memory at 32 arenas a block (GpuBatch::extract_raw launches the same)
    constexpr uint32_t A = NC == 8 ? 16u : 32u;
    __shared__ float tile[A * RP];
    const uint32_t base = blockIdx.x * A, lane = threadIdx.x;
    if (base >= n) return;
    const uint32_t count = n - base < A ? n - base : A;
    if (lane < count) extract_arena<NC>(reinterpret_cast<const ArenaState<NC>*>(states)[base + lane], tile + lane * RP);
    __syncwarp();
    float* o = out + (size_t)base * R;
    for (uint32_t k = lane; k < count * (uint32_t)R; k += A) o[k] = tile[(k / R) * RP + k % R];
}

// Every arena to a kickoff, its rng seeded from (seed, arena index).
template <int NC>
__device__ __forceinline__ void kickoff_arenas(uint32_t* states, uint32_t n, uint32_t seed) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ArenaState<NC>* a = reinterpret_cast<ArenaState<NC>*>(states) + i;
    a->rng = seed * 0x9E3779B9u + i * 0x85EBCA6Bu + 1u;
    kickoff_reset<NC>(*a);
}

// Reset-and-branch: arena i of `dst` becomes a copy of arena map[i] of `src` (the start states a trainer uploads, or a
// snapshot of live arenas; never `dst` itself), its rng reseeded from (seed, i) as at a kickoff, so branches of one
// state follow their own controls and draw their own kickoff positions after a goal. One warp per arena: the lanes copy
// its words coalesced, 16 bytes per lane when an arena is a whole number of 16-byte units (the buffer is 256-byte aligned
// from cuMemAlloc, so every arena then starts on 16 bytes). An out-of-range map entry leaves the arena as it was.
template <int NC>
__device__ __forceinline__ void branch_arenas(uint32_t* dst, uint32_t n, const uint32_t* src, uint32_t n_src,
                                              const uint32_t* map, uint32_t seed) {
    const uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) / 32u, lane = threadIdx.x % 32u;
    if (i >= n) return;
    const uint32_t m = map[i];
    if (m >= n_src) return;
    constexpr uint32_t W = sizeof(ArenaState<NC>) / 4;
    const uint32_t* s = src + (size_t)m * W;
    uint32_t* d = dst + (size_t)i * W;
    if constexpr (W % 4 == 0) {
        const uint4* s4 = reinterpret_cast<const uint4*>(s);
        uint4* d4 = reinterpret_cast<uint4*>(d);
        for (uint32_t w = lane; w < W / 4; w += 32u) d4[w] = s4[w];
    } else {
        for (uint32_t w = lane; w < W; w += 32u) d[w] = s[w];
    }
    __syncwarp();
    if (lane == 0) reinterpret_cast<ArenaState<NC>*>(d)->rng = seed * 0x9E3779B9u + i * 0x85EBCA6Bu + 1u;
}

#define RLMAX_DECIDE(NC)                                                                                       \
    extern "C" __global__ void RLMAX_TICK_BOUNDS_##NC rlmax_rollout_##NC(uint32_t* states, uint32_t n, uint32_t ticks,  \
                                                                         ArenaData ad, const uint8_t* rows,           \
                                                                         const float* table, uint32_t* stop) {        \
        rollout_arenas<NC>(states, n, ticks, ad, rows, table, stop);                                                   \
    }                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(32) rlmax_extract_##NC(const uint32_t* states, uint32_t n,           \
                                                                         float* out) {                                \
        extract_arenas<NC>(states, n, out);                                                                            \
    }                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128) rlmax_branch_##NC(uint32_t* dst, uint32_t n, const uint32_t* src, \
                                                                        uint32_t n_src, const uint32_t* map,          \
                                                                        uint32_t seed) {                              \
        branch_arenas<NC>(dst, n, src, n_src, map, seed);                                                              \
    }                                                                                                                  \
    extern "C" __global__ void RLMAX_TICK_BOUNDS_##NC rlmax_decide_##NC(uint32_t* states, uint32_t n, uint32_t ticks,   \
                                                                        ArenaData ad, const uint8_t* rows,            \
                                                                        const float* table, uint32_t* goals) {        \
        decide_arenas<NC>(states, n, ticks, ad, rows, table, goals);                                                   \
    }                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128) rlmax_kickoff_##NC(uint32_t* states, uint32_t n, uint32_t seed) { \
        kickoff_arenas<NC>(states, n, seed);                                                                           \
    }
RLMAX_DECIDE(2)
RLMAX_DECIDE(3)
RLMAX_DECIDE(4)
RLMAX_DECIDE(5)
RLMAX_DECIDE(6)
RLMAX_DECIDE(8)

// Set one render-frame fraction per arena. A null array restores the default 0.5.
template <int NC>
__device__ void set_pad_fractions(uint32_t* states, uint32_t n, const float* fractions) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) reinterpret_cast<ArenaState<NC>*>(states)[i].pad_frame_bits = fractions ? pad_encode_frac(fractions[i]) : 0u;
}
#define RLMAX_PAD_FRAC(N) extern "C" __global__ void rlmax_pad_fraction_##N(uint32_t* s, uint32_t n, const float* f) { set_pad_fractions<N>(s,n,f); }
RLMAX_PAD_FRAC(2) RLMAX_PAD_FRAC(3) RLMAX_PAD_FRAC(4) RLMAX_PAD_FRAC(5) RLMAX_PAD_FRAC(6) RLMAX_PAD_FRAC(8)

// Separate three-wheel variants: batches without Psyclops retain the ordinary tick specialization.
#define RLMAX_PSYCLOPS(NC) \
    extern "C" __global__ void RLMAX_TICK_BOUNDS_##NC rlmax_step_psyclops_##NC(uint32_t* s, uint32_t n, uint32_t t, ArenaData ad) { \
        step_arenas<NC, true>(s, n, t, ad); \
    } \
    extern "C" __global__ void RLMAX_TICK_BOUNDS_##NC rlmax_decide_psyclops_##NC(uint32_t* s, uint32_t n, uint32_t t, ArenaData ad, const uint8_t* rows, const float* table, uint32_t* out) { \
        decide_arenas<NC, true>(s, n, t, ad, rows, table, out); \
    } \
    extern "C" __global__ void RLMAX_TICK_BOUNDS_##NC rlmax_rollout_psyclops_##NC(uint32_t* s, uint32_t n, uint32_t t, ArenaData ad, const uint8_t* rows, const float* table, uint32_t* out) { \
        rollout_arenas<NC, true>(s, n, t, ad, rows, table, out); \
    }
RLMAX_PSYCLOPS(2) RLMAX_PSYCLOPS(3) RLMAX_PSYCLOPS(4)
RLMAX_PSYCLOPS(5) RLMAX_PSYCLOPS(6) RLMAX_PSYCLOPS(8)
