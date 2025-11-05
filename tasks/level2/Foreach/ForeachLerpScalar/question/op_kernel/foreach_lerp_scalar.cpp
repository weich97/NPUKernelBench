#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void foreach_lerp_scalar(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
}
