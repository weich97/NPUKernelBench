// #ifndef ARANGE_TILING_H
// #define ARANGE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(ArangeTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, totalNum);
  TILING_DATA_FIELD_DEF(uint32_t, unitNum);
  TILING_DATA_FIELD_DEF(uint32_t, unitLoops);
  TILING_DATA_FIELD_DEF(uint32_t, tailNum);
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(Arange, ArangeTilingData)
}