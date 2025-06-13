#ifndef ASCEND_KERNEL_BENCH_CUSTOM_TILING_H
#define ASCEND_KERNEL_BENCH_CUSTOM_TILING_H

#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, mode);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MseLoss, MseLossTilingData)
}

#endif  // ASCEND_KERNEL_BENCH_CUSTOM_TILING_H
