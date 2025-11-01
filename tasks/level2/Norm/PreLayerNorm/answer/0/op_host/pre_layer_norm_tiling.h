#ifndef PRE_LAYER_NORM_TILING_H
#define PRE_LAYER_NORM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PreLayerNormTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, lastDim);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PreLayerNorm, PreLayerNormTilingData)
}
#endif // PRE_LAYER_NORM_TILING_H