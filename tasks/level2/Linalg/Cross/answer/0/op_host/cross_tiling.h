#ifndef CROSS_TILING_H
#define CROSS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CrossTilingData)
  TILING_DATA_FIELD_DEF_ARR(int64_t, 128, shape);
  TILING_DATA_FIELD_DEF(int64_t, numshapes);
  TILING_DATA_FIELD_DEF(int64_t, dim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Cross, CrossTilingData)
  }
#endif // CROSS_TILING_H