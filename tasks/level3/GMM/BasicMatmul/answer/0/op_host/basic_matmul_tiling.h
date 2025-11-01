#ifndef ASCEND_KERNEL_BENCH_CUSTOM_TILING_H
#define ASCEND_KERNEL_BENCH_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, n);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BasicMatmul, TilingData)
} // namespace optiling

#endif