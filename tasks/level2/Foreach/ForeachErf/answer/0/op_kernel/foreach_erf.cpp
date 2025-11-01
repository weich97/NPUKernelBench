/*!
 * \file foreach_erf.cpp
 * \brief
 */

 #include "kernel_operator.h"
 #include "lib/math/kernel_operator_asin_intf.h"
 
 // op kernel building at build_out directory, it's not fully aligned with source code structure
 // current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_triangle.h"
 
using namespace AscendC;
using namespace Common::OpKernel;

extern "C" __global__ __aicore__ void foreach_erf(GM_ADDR x,  GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachTriangle<half, half, Erf<half, false>> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachTriangle<float, float, Erf<float, false>> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        ForeachTriangle<bfloat16_t, float, Erf<float, false>> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
}
