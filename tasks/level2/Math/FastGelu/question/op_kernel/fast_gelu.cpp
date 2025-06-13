#include "kernel_operator.h"
using namespace AscendC;


extern "C" __global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
}