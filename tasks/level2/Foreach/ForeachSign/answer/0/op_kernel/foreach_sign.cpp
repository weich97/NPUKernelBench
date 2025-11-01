/*!
 * \file foreach_sign.cpp
 * \brief
 */

 #include "foreach_sign.h"

 using namespace ForeachSign;
 
 extern "C" __global__ __aicore__ void foreach_sign(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, 
     GM_ADDR tiling) {
     GET_TILING_DATA(tilingData, tiling);
 
     //foreach(vector) not need workspace
     GM_ADDR userWS = nullptr;
 
     if (TILING_KEY_IS(1)) {
         ForeachSignND<half> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(2)) {
         ForeachSignND<float> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(3)) {
         ForeachSignND<int32_t> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(4)) {
         ForeachSignND<bfloat16_t> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(7)) {
         ForeachSignND<int8_t> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     } else if (TILING_KEY_IS(10)) {
         ForeachSignND<int64_t> op;
         op.Init(x, y, userWS, &tilingData);
         op.Process();
     }
 }