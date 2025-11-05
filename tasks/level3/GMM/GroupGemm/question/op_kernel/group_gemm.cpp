#include <kernel_operator.h>
using namespace AscendC;
extern "C" __global__ __aicore__ void group_gemm(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR beta,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
}