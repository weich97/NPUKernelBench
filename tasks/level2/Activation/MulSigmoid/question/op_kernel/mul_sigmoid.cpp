#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void mul_sigmoid(GM_ADDR x1, GM_ADDR x2, GM_ADDR out_buf, GM_ADDR workspace, GM_ADDR tiling)
{
}