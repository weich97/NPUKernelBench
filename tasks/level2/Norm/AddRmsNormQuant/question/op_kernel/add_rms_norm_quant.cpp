#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void add_rms_norm_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1,
    GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR workspace,
    GM_ADDR tiling)
{
    
}