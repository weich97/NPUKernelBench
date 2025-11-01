#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void layer_norm_v4(GM_ADDR x, GM_ADDR normalized_shape, GM_ADDR gamma, GM_ADDR beta,
    GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
}
