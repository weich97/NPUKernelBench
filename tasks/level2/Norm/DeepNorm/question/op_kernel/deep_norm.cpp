#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void deep_norm(GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean,
    GM_ADDR rstd, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
}