#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;
struct ForeachNormCompileInfo {
    uint32_t coreNum;
};
BEGIN_TILING_DATA_DEF(ForeachReduceTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, inputsTensorUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, totalTensorCount);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 256, tensorDataCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 64, tensorStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 64, tensorEndList);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 64, tensorStartOffsetList);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 64, tensorEndOffsetList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 256, tensorMiddleCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 256, tensorMiddleStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, 64, coreMiddleOffsetList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ForeachNorm, ForeachReduceTilingData)
}