#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void dequant_swiglu_quant(GM_ADDR xGM, GM_ADDR weightSscaleGM,
                                                           GM_ADDR activationScaleGM, GM_ADDR biasGM,
                                                           GM_ADDR quantScaleGM, GM_ADDR quantOffsetGM,
                                                           GM_ADDR groupIndex, GM_ADDR yGM, GM_ADDR scaleGM,
                                                           GM_ADDR workspace, GM_ADDR tiling) {

}