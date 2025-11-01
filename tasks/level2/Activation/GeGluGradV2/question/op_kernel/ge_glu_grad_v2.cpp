#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void ge_glu_grad_v2(GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx, GM_ADDR workspace,
                                                     GM_ADDR tiling) 
                                                     {
                                                        
                                                     }