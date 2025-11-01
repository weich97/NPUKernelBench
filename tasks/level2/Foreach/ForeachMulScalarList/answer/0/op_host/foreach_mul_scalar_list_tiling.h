#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_CONT = 50;
constexpr uint16_t MAX_CORE_CONT = 50;
struct ForeachCompileInfo {
    uint64_t coreNum;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0ASize;
    uint64_t l0BSize;
    uint64_t l0CSize;
    uint64_t sysWorkspaceSize;
};

BEGIN_TILING_DATA_DEF(ForeachCommonTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, inputsTensorUbSize);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_CONT, tensorDataCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorEndList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorStartOffsetList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorEndOffsetList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ForeachMulScalarList, ForeachCommonTilingData)
}