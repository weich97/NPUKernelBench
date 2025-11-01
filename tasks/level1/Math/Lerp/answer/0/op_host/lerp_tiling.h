#ifndef LERP_TILING_H
#define LERP_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
  BEGIN_TILING_DATA_DEF(LerpTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, total_length);
  TILING_DATA_FIELD_DEF(uint32_t, start_length);
  TILING_DATA_FIELD_DEF(uint32_t, end_length);
  TILING_DATA_FIELD_DEF(uint32_t, weight_length);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(uint32_t, mode);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, shape);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce1);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce2);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, reduce3);
  TILING_DATA_FIELD_DEF(uint32_t, dim);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Lerp, LerpTilingData)
}
#endif // LERP_TILING_H