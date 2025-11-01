#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void cross_entropy_loss_grad(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target,
                                                              GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
                                                              GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling)
{
}