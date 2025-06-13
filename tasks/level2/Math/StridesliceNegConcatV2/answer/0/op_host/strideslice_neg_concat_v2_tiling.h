#ifndef STRIDESLICE_NEG_CONCAT_V2_TILING_H
#define STRIDESLICE_NEG_CONCAT_V2_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(StridesliceNegConcatV2Tiling)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumAverage);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumLast);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StridesliceNegConcatV2, StridesliceNegConcatV2Tiling)
}
#endif // STRIDESLICE_NEG_CONCAT_V2_TILING_H