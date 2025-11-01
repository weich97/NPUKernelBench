#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void batch_norm_v3(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR mean,
    GM_ADDR variance, GM_ADDR y, GM_ADDR mean_out, GM_ADDR variance_out, GM_ADDR save_mean, GM_ADDR save_var,
    GM_ADDR workspace, GM_ADDR tiling) {
}
