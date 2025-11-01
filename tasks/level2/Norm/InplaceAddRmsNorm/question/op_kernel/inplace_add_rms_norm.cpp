#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void inplace_add_rms_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    
}