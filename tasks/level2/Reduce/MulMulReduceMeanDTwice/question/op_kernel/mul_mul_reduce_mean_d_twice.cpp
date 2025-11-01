#include "kernel_operator.h"
using namespace AscendC;


extern "C" __global__ __aicore__ void mul_mul_reduce_mean_d_twice(GM_ADDR mul0_input0, GM_ADDR mul0_input1, GM_ADDR mul1_input0, GM_ADDR add_y, GM_ADDR gamma, GM_ADDR beta, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
}