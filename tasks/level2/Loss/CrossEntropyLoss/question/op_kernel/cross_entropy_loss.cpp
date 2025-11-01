#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void cross_entropy_loss(GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob, 
                                                         GM_ADDR zloss, GM_ADDR lse_for_zloss, GM_ADDR workspace, GM_ADDR tiling) 
                                                          {
}