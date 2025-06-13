#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void swi_glu_grad(GM_ADDR gradout_gm, GM_ADDR input_gm, GM_ADDR output_gm,
  GM_ADDR workspace, GM_ADDR tiling) {
}