#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void group_norm_swish(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                                      GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
}
