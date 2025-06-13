#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void foreach_mul_list(GM_ADDR inputs_1, GM_ADDR inputs_2,
    GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
}