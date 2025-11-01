#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void complex_mat_dot(GM_ADDR matx, GM_ADDR maty,
                                                    GM_ADDR result, GM_ADDR workspace, GM_ADDR tilingGm)
{
}