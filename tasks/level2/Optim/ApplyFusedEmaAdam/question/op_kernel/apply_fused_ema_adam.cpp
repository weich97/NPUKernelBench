#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void apply_fused_ema_adam(GM_ADDR grad, GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR s,
                                                           GM_ADDR step, GM_ADDR var_ref, GM_ADDR m_ref, GM_ADDR v_ref,
                                                           GM_ADDR s_ref, GM_ADDR workspace, GM_ADDR tiling) {

}