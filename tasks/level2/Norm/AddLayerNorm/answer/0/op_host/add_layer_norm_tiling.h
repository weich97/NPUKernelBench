#ifndef ASCEND_KERNEL_BENCH_CUSTOM_TILING_H
#define ASCEND_KERNEL_BENCH_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddLayerNormTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numCore);
TILING_DATA_FIELD_DEF(uint32_t, numLastDim);
TILING_DATA_FIELD_DEF(uint32_t, numFirstDim);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerCore);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerCoreTail);
TILING_DATA_FIELD_DEF(uint32_t, firstDimPerTime);
TILING_DATA_FIELD_DEF(uint32_t, lastDimPerTime);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, aveFactor);
TILING_DATA_FIELD_DEF(uint32_t, colMoveCnt);
TILING_DATA_FIELD_DEF(uint32_t, colTail);
TILING_DATA_FIELD_DEF(uint32_t, workspaceSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddLayerNorm, AddLayerNormTilingData)
}  // namespace optiling

#endif  // ASCEND_KERNEL_BENCH_CUSTOM_TILING_H
