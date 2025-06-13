#ifndef ADD_SIGMOID_MUL_REDUCE_SUM_D_TILING_H
#define ADD_SIGMOID_MUL_REDUCE_SUM_D_TILING_H
#include "register/tilingdata_base.h"

namespace optiling{
    BEGIN_TILING_DATA_DEF(AddSigmoidMulReduceSumDTilingData)
    TILING_DATA_FIELD_DEF(int32_t, formerCoreNum);
    TILING_DATA_FIELD_DEF(int32_t, formerCoreLength);
    TILING_DATA_FIELD_DEF(int32_t, formerTileNum);
    TILING_DATA_FIELD_DEF(int32_t, formerTileLength);
    TILING_DATA_FIELD_DEF(int32_t, formerLastTileLength);
    TILING_DATA_FIELD_DEF(int32_t, tailCoreNum);
    TILING_DATA_FIELD_DEF(int32_t, tailCoreLength);
    TILING_DATA_FIELD_DEF(int32_t, tailTileNum);
    TILING_DATA_FIELD_DEF(int32_t, tailTileLength);
    TILING_DATA_FIELD_DEF(int32_t, tailLastTileLength);
    TILING_DATA_FIELD_DEF(int32_t, addInput0Dim1234Length);
    TILING_DATA_FIELD_DEF(int32_t, addInput0Dim14Length);
    TILING_DATA_FIELD_DEF(int32_t, addInput0Dim23Length);
    TILING_DATA_FIELD_DEF(int32_t, addInput0Dim1Length);
    TILING_DATA_FIELD_DEF(int32_t, addInput0Dim234Length);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(AddSigmoidMulReduceSumD, AddSigmoidMulReduceSumDTilingData)
}
#endif // ADD_SIGMOID_MUL_REDUCE_SUM_D_TILING_H