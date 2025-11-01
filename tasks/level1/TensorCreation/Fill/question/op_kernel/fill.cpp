#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void fill(GM_ADDR dims, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
}