#include "kernel_operator.h"
using namespace AscendC;

// 核函数入口
extern "C" __global__ __aicore__ void strideslice_neg_concat_v2(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) 
{
}