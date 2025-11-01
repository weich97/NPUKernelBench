#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void inplace_fused_matmul_softmax_grad(
    GM_ADDR softmaxOutput, 
    GM_ADDR gradOutput, 
    GM_ADDR values, 
    GM_ADDR workspace, 
    GM_ADDR tiling
) {
    
}