#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void muls( GM_ADDR x,
      GM_ADDR value,
      GM_ADDR y, 
      GM_ADDR workspace, 
      GM_ADDR tiling)
{
}