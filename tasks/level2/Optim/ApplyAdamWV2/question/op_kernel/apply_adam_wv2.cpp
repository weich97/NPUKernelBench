#include "kernel_operator.h"

using namespace AscendC;


// 核函数入口
extern "C" __global__ __aicore__ void apply_adam_wv2(GM_ADDR var_ref, GM_ADDR m_ref, GM_ADDR v_ref,
    GM_ADDR grad, GM_ADDR step, GM_ADDR maxGradNorm, GM_ADDR workspace, GM_ADDR tiling) {
}