#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void deep_norm_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR gx, GM_ADDR gamma, GM_ADDR mean,
    GM_ADDR rstd, GM_ADDR dx, GM_ADDR dgx, GM_ADDR dbeta, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling)
{
}