#include "kernel_operator.h"

using namespace AscendC;

// Implementation note.
extern "C" __global__ __aicore__ void rms_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
}