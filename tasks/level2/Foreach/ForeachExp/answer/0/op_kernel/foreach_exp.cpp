/*!
 * \file foreach_exp.cpp
 * \brief
 */

 #include "kernel_operator.h"
 #include "lib/math/kernel_operator_asin_intf.h"
 
 // op kernel building at build_out directory, it's not fully aligned with source code structure
 // current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_implict_output.h"
 
using namespace AscendC;
using namespace Common::OpKernel;

extern "C" __global__ __aicore__ void foreach_exp(GM_ADDR x,  GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachImplictOutput<half, half, Exp, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachImplictOutput<float, float, Exp, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } 
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachImplictOutput<bfloat16_t, float, Exp, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
#endif
}