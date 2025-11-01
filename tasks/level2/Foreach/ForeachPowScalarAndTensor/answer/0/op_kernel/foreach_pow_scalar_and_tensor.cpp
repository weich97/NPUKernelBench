 /*!
 * \file foreach_pow_scalar_and_tensor.cpp
 * \brief
 */

#include "foreach_pow_scalar_and_tensor.h"

using namespace ForeachPowScalarAndTensor;

extern "C" __global__ __aicore__ void foreach_pow_scalar_and_tensor(GM_ADDR scalar,
    GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachPowScalarAndTensorND<half> op;
        op.Init(scalar, inputs, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachPowScalarAndTensorND<float> op;
        op.Init(scalar, inputs, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ForeachPowScalarAndTensorND<int> op;
        op.Init(scalar, inputs, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        ForeachPowScalarAndTensorND<bfloat16_t> op;
        op.Init(scalar, inputs, outputs, userWS, &tilingData);
        op.Process();
    }
}
