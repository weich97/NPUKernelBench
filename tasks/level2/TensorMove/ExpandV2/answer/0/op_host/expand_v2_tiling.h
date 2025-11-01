#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ExpandV2TilingData)
  TILING_DATA_FIELD_DEF(uint64_t, expandSize);
  TILING_DATA_FIELD_DEF(uint64_t, blockLength);
  TILING_DATA_FIELD_DEF(uint64_t, tileNum);
  TILING_DATA_FIELD_DEF(uint64_t, tileLength);
  TILING_DATA_FIELD_DEF(uint64_t, miniTileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ExpandV2, ExpandV2TilingData)
}
