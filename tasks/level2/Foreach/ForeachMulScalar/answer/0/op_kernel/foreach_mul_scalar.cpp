/*!
 * \file foreach_mul_scalar.cpp
 * \brief
 */

#include "kernel_operator.h"
// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_one_scalar_binary.h"

using namespace AscendC;
using namespace Common::OpKernel;

extern "C" __global__ __aicore__ void foreach_mul_scalar(GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace,
    GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
    ForeachOneScalarBinary<half, half, Muls, 1> op;
    op.Init(inputs, scalar, outputs, userWS, &tilingData);
    op.Process();
    } else if (TILING_KEY_IS(2)) {
    ForeachOneScalarBinary<float, float, Muls, 1> op;
    op.Init(inputs, scalar, outputs, userWS, &tilingData);
    op.Process();
    }
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(3)) {
    ForeachOneScalarBinary<int, int, Muls, 1> op;
    op.Init(inputs, scalar, outputs, userWS, &tilingData);
    op.Process();
    } else if (TILING_KEY_IS(4)) {
    ForeachOneScalarBinary<bfloat16_t, float, Muls, 1> op;
    op.Init(inputs, scalar, outputs, userWS, &tilingData);
    op.Process();
    }
#endif
}
