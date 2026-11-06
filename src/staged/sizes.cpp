#include "staged_tick.h"
static_assert(sizeof(rlmax::StagedTick<2>) % 4 == 0 && sizeof(rlmax::StagedTick<4>) % 4 == 0 && sizeof(rlmax::StagedTick<6>) % 4 == 0, "stage scratch must be whole words");
extern "C" unsigned int rlmax_staged_words(unsigned int n) {
    switch(n) {
    case 2: return sizeof(rlmax::StagedTick<2>)/sizeof(unsigned int);
    case 4: return sizeof(rlmax::StagedTick<4>)/sizeof(unsigned int);
    case 6: return sizeof(rlmax::StagedTick<6>)/sizeof(unsigned int);
    default: return 0;
    }
}
