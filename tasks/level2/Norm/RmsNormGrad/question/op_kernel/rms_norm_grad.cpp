#include "kernel_operator.h"

using namespace AscendC;

// Implementation note.
extern "C" __global__ __aicore__ void rms_norm_grad(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling)
{
}