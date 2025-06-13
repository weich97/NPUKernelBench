#ifndef ADDCDIV_TILING_H
#define ADDCDIV_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddcdivTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalBigTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Addcdiv, AddcdivTilingData)
} // namespace optiling
#endif // ADDCDIV_TILING_H
