/*!
 * \file foreach_neg.cpp
 * \brief
 */

 #include "kernel_operator.h" 
 // op kernel building at build_out directory, it's not fully aligned with source code structure
 // current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_implict_output_level_zero_api.h"
 
using namespace AscendC;
using namespace Common::OpKernel;

extern "C" __global__ __aicore__ void foreach_neg(GM_ADDR x,  GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachImplictOutputLevelZeroApi<half, half, Sub, 2, 1> op;
        op.Init(x, y, userWS, &tilingData, half(0));
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachImplictOutputLevelZeroApi<float, float, Sub, 2, 1> op;
        op.Init(x, y, userWS, &tilingData, float(0));
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ForeachImplictOutputLevelZeroApi<int, int, Sub, 2, 1> op;
        op.Init(x, y, userWS, &tilingData, 0);
        op.Process();
    } 
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachImplictOutputLevelZeroApi<bfloat16_t, float, Sub, 2, 1> op;
        op.Init(x, y, userWS, &tilingData, float(0));
        op.Process();
    }
#endif
}