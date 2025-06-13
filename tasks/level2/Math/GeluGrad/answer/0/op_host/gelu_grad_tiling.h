#ifndef ASCEND_KERNEL_BENCH_CUSTOM_TILING_H
#define ASCEND_KERNEL_BENCH_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluGradTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalBigTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);
  TILING_DATA_FIELD_DEF(uint32_t, versionNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluGrad, GeluGradTilingData)
}

#endif  // ASCEND_KERNEL_BENCH_CUSTOM_TILING_H
