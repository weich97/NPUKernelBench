#include "kernel_operator.h"

using namespace AscendC;

// Implementation note.
extern "C" __global__ __aicore__ void foreach_non_finite_check_and_unscale(GM_ADDR scaled_grads, GM_ADDR found_inf,
                                                                           GM_ADDR inv_scale, GM_ADDR workspace,
                                                                           GM_ADDR tiling) {
}