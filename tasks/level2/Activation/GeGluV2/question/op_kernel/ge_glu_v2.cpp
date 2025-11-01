#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void ge_glu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR gelu,
                                                GM_ADDR workspace, GM_ADDR tiling) 
{

}