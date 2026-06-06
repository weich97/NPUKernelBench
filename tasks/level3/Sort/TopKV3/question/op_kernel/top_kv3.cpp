#include "kernel_operator.h"

using namespace AscendC;


// Implementation note.
extern "C" __global__ __aicore__ void top_kv3(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
}