/*!
 * \file foreach_abs.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "foreach_triangle.h"
 
using namespace AscendC;
using namespace Common::OpKernel;
 
template <typename T>
__aicore__ void AbsAdapter(
    const LocalTensor<T>& dstLocal, 
    const LocalTensor<T>& srcLocal, 
    const uint32_t uValue) {
    Abs(dstLocal, srcLocal, static_cast<int32_t>(uValue));
}
 
extern "C" __global__ __aicore__ void foreach_abs(
    GM_ADDR x, 
    GM_ADDR y, 
    GM_ADDR workspace, 
    GM_ADDR tiling) {
    
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachTriangle<half, half, AbsAdapter<half>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachTriangle<float, float, AbsAdapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } 
    #if __CCE_AICORE__ == 220
        else if (TILING_KEY_IS(4)) {
            ForeachTriangle<bfloat16_t, float, AbsAdapter<float>, 2, 1> op;
            op.Init(x, y, userWS, &tilingData);
            op.Process();
    }
    #endif
}
 