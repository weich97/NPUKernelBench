#ifndef ASCEND_KERNEL_BENCH_CUSTOM_TILING_H
#define ASCEND_KERNEL_BENCH_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_CONT = 50;
constexpr uint16_t MAX_CORE_CONT = 50;

BEGIN_TILING_DATA_DEF(ForeachCommonTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, inputsTensorUbSize);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_CONT, tensorDataCountList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorStartList);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorEndList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorStartOffsetList);
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_CONT, tensorEndOffsetList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ForeachLog, ForeachCommonTilingData)
}  // namespace optiling

#endif  // ASCEND_KERNEL_BENCH_CUSTOM_TILING_H
