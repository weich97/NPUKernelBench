#include "register/tilingdata_base.h"

namespace optiling {
constexpr size_t MAX_DIM_NUM = 8;
BEGIN_TILING_DATA_DEF(StridedSliceAssignV2TilingData)
    TILING_DATA_FIELD_DEF(int64_t, dimNum);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, varDim);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, inputValueDim);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, begin);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, strides);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, varCumShape);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_DIM_NUM, inputCumShape);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StridedSliceAssignV2, StridedSliceAssignV2TilingData)

struct StridedSliceAssignV2CompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
};
}  // namespace optiling