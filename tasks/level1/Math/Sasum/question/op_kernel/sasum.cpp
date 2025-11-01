#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void sasum(GM_ADDR inGM, GM_ADDR outGM,
                                            GM_ADDR workspace, GM_ADDR tilingGM)
{
}