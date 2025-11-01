#ifndef ASCEND_KERNEL_BENCH_CUSTOM_TILING_H
#define ASCEND_KERNEL_BENCH_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
static constexpr uint32_t MAX_TENSOR_COUNT = 32;

BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, groupCount);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_COUNT, mList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_COUNT, kList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_COUNT, nList);
TILING_DATA_FIELD_DEF(uint32_t, align);
TILING_DATA_FIELD_DEF(uint32_t, paddingASize);
TILING_DATA_FIELD_DEF(uint32_t, paddingBSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupGemm, TilingData)
} // namespace optiling

#endif