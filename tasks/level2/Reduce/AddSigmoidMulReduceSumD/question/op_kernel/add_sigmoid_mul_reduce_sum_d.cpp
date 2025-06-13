#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void
add_sigmoid_mul_reduce_sum_d(GM_ADDR add_0_input0, GM_ADDR add_0_input1, GM_ADDR mul_0_input1, GM_ADDR mult_1_input1, GM_ADDR mult_2_input1, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    
}