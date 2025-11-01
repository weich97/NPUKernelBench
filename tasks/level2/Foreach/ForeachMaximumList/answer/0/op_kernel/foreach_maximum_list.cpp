/*!
 * \file foreach_maximum_list.cpp
 * \brief
 */

#include "kernel_operator.h"

// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_no_scalar_binary.h"

using namespace AscendC;
using namespace Common::OpKernel;

extern "C" __global__ __aicore__ void foreach_maximum_list(GM_ADDR inputs_1, GM_ADDR inputs_2,
    GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachNoScalarBinary<half, half, Max> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachNoScalarBinary<float, float, Max> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ForeachNoScalarBinary<int, int, Max> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    }  
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachNoScalarBinary<bfloat16_t, float, Max> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    }
#endif
}