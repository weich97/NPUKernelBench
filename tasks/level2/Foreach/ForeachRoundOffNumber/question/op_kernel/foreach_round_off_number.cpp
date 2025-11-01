#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void foreach_round_off_number(
    GM_ADDR x, GM_ADDR roundMode, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
}
