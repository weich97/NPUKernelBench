#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void snrm2(GM_ADDR x, GM_ADDR result,
                                        GM_ADDR workSpace, GM_ADDR tilingGm)
{
}