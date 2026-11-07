// The round countdown. While it runs the game keeps every car's bDriving off: the controls read as zero, the jump button
// kept as the raw input the next press edge needs, the throttle kept for the gates it feeds (BTN_UNDRIVEN, state.h).
// The game ends it with two timers fed by the render frames' DeltaSeconds: a first second (phase 1), then three more
// seconds, each fired when the counter passes 1 (phase 2, fires 1..3). The counter is a float and is added once per
// render frame, so the game's release tick follows from the float sums, not from a tick count.
//
// RLMAX has one frame model, pads.h: every tick ends a render frame at the arena's fraction f (a negative f ends none).
// That frame lasts (1 + f) - f ticks, the interpolation fraction's change added to the whole tick, as the game's frame
// time does when the previous frame ended at the same fraction. For the default f = 0.5 that is exactly one tick.
// The tick that raises the last fire is still held: the countdown ends after it, so the next tick drives.
//
// A state with `countdown == 0` is the default: no countdown, nothing here costs more than the test.
#pragma once

#include "consts.h"
#include "pads.h"
#include "state.h"

namespace rlmax {

constexpr uint32_t COUNTDOWN_PHASE_MASK = 3u;
constexpr uint32_t COUNTDOWN_FIRES_SHIFT = 2;

// A running countdown: the state it starts from (phase 1, fires 0, counter 0) is what a round reset leaves.
RLMAX_HD uint32_t countdown_word(uint32_t phase, uint32_t fires) { return phase | fires << COUNTDOWN_FIRES_SHIFT; }

// Start of the tick: every car undriven. Throttle stays; steering, air control and the buttons go, the jump button's state stays as JUMP_HELD.
template <int NC>
RLMAX_HD void countdown_hold(ArenaState<NC>& s) {
    for (int i = 0; i < NC; i++) {
        Controls& ctl = s.cars[i].ctl;
        ctl = Controls{ctl.throttle, 0.0f, 0.0f, 0.0f, 0.0f, BTN_UNDRIVEN | ((ctl.buttons & (BTN_JUMP | BTN_JUMP_HELD)) ? BTN_JUMP_HELD : 0u)};
    }
}

// End of the tick: the frame's DeltaSeconds into the counter, and the phase it moves.
RLMAX_HD void countdown_advance(uint32_t& word, float& count, uint32_t pad_frame_bits) {
    const float f = pad_decode_frac(pad_frame_bits);
    if (f < 0.0f) return;  // no frame ended this tick
    count += ((1.0f + f) - f) * c::DT;
    if (!(count > 1.0f)) return;
    if ((word & COUNTDOWN_PHASE_MASK) == 1u) {
        word = countdown_word(2u, 0u);
        count = 0.0f;
        return;
    }
    count -= 1.0f;
    const uint32_t fires = (word >> COUNTDOWN_FIRES_SHIFT) + 1u;
    if (fires == 3u) {
        word = 0u;
        count = 0.0f;
    } else {
        word = countdown_word(2u, fires);
    }
}

}  // namespace rlmax
