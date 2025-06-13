#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void mse_loss(GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
}