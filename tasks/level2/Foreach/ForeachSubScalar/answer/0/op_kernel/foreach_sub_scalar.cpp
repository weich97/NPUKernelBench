/*!
 * \file foreach_sub_scalar.cpp
 * \brief
 */

#include "kernel_operator.h"
// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_one_scalar_binary_level_zero_api.h"

using namespace AscendC;
using namespace Common::OpKernel;

extern "C" __global__ __aicore__ void foreach_sub_scalar(GM_ADDR x, GM_ADDR scalar, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {

    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
    ForeachOneScalarBinaryLevelZeroApi<half, float, Sub, 2, 1> op;
    op.Init(x, scalar, y, userWS, &tilingData);
    op.Process();
    } else if (TILING_KEY_IS(2)) {
    ForeachOneScalarBinaryLevelZeroApi<float, float, Sub, 2, 1> op;
    op.Init(x, scalar, y, userWS, &tilingData);
    op.Process();
    } else if (TILING_KEY_IS(3)) {
    ForeachOneScalarBinaryLevelZeroApi<int, int, Sub, 2, 1> op;
    op.Init(x, scalar, y, userWS, &tilingData);
    op.Process();
    }
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
    ForeachOneScalarBinaryLevelZeroApi<bfloat16_t, float, Sub, 2, 1> op;
    op.Init(x, scalar, y, userWS, &tilingData);
    op.Process();
    }
#endif
}