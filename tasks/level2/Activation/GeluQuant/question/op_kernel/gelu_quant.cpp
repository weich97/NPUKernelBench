#include "kernel_operator.h"

using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void gelu_quant(GM_ADDR x, GM_ADDR input_scale, GM_ADDR input_offset, GM_ADDR y,
    GM_ADDR out_scale, GM_ADDR workspace, GM_ADDR tiling_data)
{
}