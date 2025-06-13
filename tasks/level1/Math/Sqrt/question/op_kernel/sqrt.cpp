#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
}