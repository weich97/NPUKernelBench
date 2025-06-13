#ifndef EQUAL_TILING_H
#define EQUAL_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EqualTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumMean);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumEnd);
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthMean);
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthEnd);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthMean);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthEnd);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Equal, EqualTilingData)
}
#endif // EQUALCUSTOM_TILING_H