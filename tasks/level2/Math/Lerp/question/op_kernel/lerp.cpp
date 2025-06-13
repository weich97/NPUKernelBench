#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void lerp(GM_ADDR start,
                                           GM_ADDR end,
                                           GM_ADDR weight,
                                           GM_ADDR y,
                                           GM_ADDR workspace,
                                           GM_ADDR tiling) {
   
}