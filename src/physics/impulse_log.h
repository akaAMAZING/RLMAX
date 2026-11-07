// Debug capture of each car's per-source velocity deltas for one tick (lead 56: the impulse audit against the game's own
// impulse log). Compiled only into the CPU build with RLMAX_IMPULSE_LOG (cargo feature `impulse_log` of the rlmax
// crate); without it every macro below is its bare expression, so the GPU kernels and the normal CPU build are the
// same code.
//
// Units are the game's records': the linear delta is a BT velocity change (velocity-space, impulse times inverse mass
// where the call is massed), the angular delta the angular velocity change (rad/s). Entries are summed per
// (source, accumulated or applied directly), as the game logs one record per source and kind.
#pragma once

#include <string.h>

#include "consts.h"

namespace rlmax {

// RLMAX source -> the game's ImpulseRecordType (cpp_records.rs):
//   IL_SUSP     update_suspension's add_rel (spring + damping + the extra pushback)       WheelsSuspension
//   IL_FRIC     apply_friction_impulses' add_rel                                           WheelsFriction
//   IL_STICKY   the sticky force, plus the re-drive delta after a dodge or double jump    StickyForce
//   IL_JUMP     the jump's immediate impulse and its held acceleration                     Jump
//   IL_DJUMP    the double jump's impulse                                                  DoubleJump
//   IL_DODGE    the dodge's linear impulse                                                 DodgeImpulse
//   IL_DTORQUE  the flip's air torque, and the flip's Z damping (a direct velocity change; the game logs it in
//               DodgeTorque's lin, tick.h)                                                  DodgeTorque
//   IL_AIR      air control torque and damping, and the air throttle                       AirControl
//   IL_BOOST    the boost acceleration                                                     Boost
//   IL_AUTOROLL auto-roll force and torque                                                 (no game type known)
//   IL_AUTOFLIP the auto-flip impulse (its ang-vel kick is applied directly, not captured) (no game type known)
enum ImpSource { IL_SUSP, IL_FRIC, IL_STICKY, IL_JUMP, IL_DJUMP, IL_DODGE, IL_DTORQUE, IL_AIR, IL_BOOST, IL_AUTOROLL, IL_AUTOFLIP, IL_N };

#ifdef RLMAX_IMPULSE_LOG

struct ImpEntry {
    float lin[3], ang[3];
    float n;  // number of adds summed
};
struct ImpLog {
    ImpEntry e[IL_N][2];  // [source][accumulated]
    ImpEntry w[2][4];     // [suspension, friction][wheel]
};
struct ImpLast {
    V3 lin, ang;
    bool accum;
};
inline thread_local ImpLog g_imp_log[8];
inline thread_local ImpLast g_imp_last;
inline thread_local float g_wheel_log[8][4][12];
inline thread_local M3 g_seed_basis[9];
inline thread_local bool g_seed_valid[9];

inline void imp_clear() { memset(g_imp_log, 0, sizeof(g_imp_log)); }
inline void imp_add(ImpEntry& d, const ImpLast& l) {
    d.lin[0] += l.lin.x; d.lin[1] += l.lin.y; d.lin[2] += l.lin.z;
    d.ang[0] += l.ang.x; d.ang[1] += l.ang.y; d.ang[2] += l.ang.z;
    d.n += 1.0f;
}
inline void imp_rec(int ci, int src, int wheel) {
    ImpLog& g = g_imp_log[ci];
    imp_add(g.e[src][g_imp_last.accum ? 1 : 0], g_imp_last);
    if (wheel >= 0) imp_add(g.w[src == IL_FRIC ? 1 : 0][wheel], g_imp_last);
}
#define RLMAX_IMP(ci, src, expr) do { expr; ::rlmax::imp_rec((ci), (src), -1); } while (0)
// the ball's seed slot follows the cars' (0..NC-1): 7 for up to seven cars, 8 for eight
#define RLMAX_SEED_BALL(b, keep) do { const int slot_ = NC == 8 ? 8 : 7; if (::rlmax::g_seed_valid[slot_]) { (b).r = ::rlmax::g_seed_basis[slot_]; (b).inv_i = world_inv_inertia((b).r, (b).inv_i_local); if (!(keep)) ::rlmax::g_seed_valid[slot_] = false; } } while (0)
#define RLMAX_SEED_BASIS(ci, b) do { if (::rlmax::g_seed_valid[ci]) { (b).r = ::rlmax::g_seed_basis[ci]; (b).inv_i = world_inv_inertia((b).r, (b).inv_i_local); ::rlmax::g_seed_valid[ci] = false; } } while (0)
#define RLMAX_WHEEL_DBG(ci, i, w) do { float* d_ = ::rlmax::g_wheel_log[ci][i]; d_[0] = (w).hit ? 1.0f : 0.0f; d_[1] = (w).susp_len; d_[2] = (w).susp_rel_vel; d_[3] = (w).contact_normal.x; d_[4] = (w).contact_normal.y; d_[5] = (w).contact_normal.z; d_[6] = (w).steer_angle; d_[7] = (w).engine_force; d_[8] = (w).brake; d_[9] = (w).lat_friction; d_[10] = (w).long_friction; d_[11] = (w).extra_pushback; } while (0)
#define RLMAX_IMPW(ci, src, wheel, expr) do { expr; ::rlmax::imp_rec((ci), (src), (wheel)); } while (0)
#define RLMAX_IMP_LAST(lin_, ang_, accum_) (::rlmax::g_imp_last = ::rlmax::ImpLast{(lin_), (ang_), (accum_)})
// a statement that changes the body's vertical velocity directly (the flip's Z damping), logged as its velocity delta
#define RLMAX_IMP_VZ(ci, src, body, stmt) do { const float vz0_ = (body).vel.z; stmt;     RLMAX_IMP_LAST(v3(0.0f, 0.0f, (body).vel.z - vz0_), v3s(0.0f), false); ::rlmax::imp_rec((ci), (src), -1); } while (0)

#else

#define RLMAX_IMP(ci, src, expr) expr
#define RLMAX_SEED_BASIS(ci, b) ((void)0)
#define RLMAX_SEED_BALL(b, keep) ((void)0)
#define RLMAX_WHEEL_DBG(ci, i, w) ((void)0)
#define RLMAX_IMPW(ci, src, wheel, expr) expr
#define RLMAX_IMP_LAST(lin_, ang_, accum_) ((void)0)
#define RLMAX_IMP_VZ(ci, src, body, stmt) stmt

#endif

}  // namespace rlmax
