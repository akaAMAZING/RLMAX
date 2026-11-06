// Test-only recorded respawn overrides (cargo feature `full_state`, CPU only).
#pragma once

#include "math.h"

namespace rlmax {

#ifdef RLMAX_FULL_STATE
inline thread_local float g_fs_spawn_z[8];  // a car's recorded CarConfig.spawn_z, uu; the game's cars sharing a preset differ
inline thread_local uint8_t g_fs_respawn[8];  // a demolished car's respawn spot as recorded, consumed at its respawn
#endif

}  // namespace rlmax
