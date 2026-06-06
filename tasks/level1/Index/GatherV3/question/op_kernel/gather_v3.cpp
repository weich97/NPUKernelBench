#include "kernel_operator.h"

using namespace AscendC;


  // Alignment and tail-handling logic.
extern "C" __global__ __aicore__ void gather_v3(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
}
