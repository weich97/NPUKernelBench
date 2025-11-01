#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void add_layer_norm_grad(GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd,
    GM_ADDR mean, GM_ADDR gamma, GM_ADDR dsum, GM_ADDR d_x, GM_ADDR d_gamma, GM_ADDR d_beta, GM_ADDR workspace,
    GM_ADDR tiling) {
}
