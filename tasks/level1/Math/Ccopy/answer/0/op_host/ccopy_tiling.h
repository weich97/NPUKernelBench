#ifndef CCOPY_TILING_H
#define CCOPY_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {

constexpr static int MAX_ARRAY_NUM = 48;

BEGIN_TILING_DATA_DEF(CcopyTilingData)
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, startOffset);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_ARRAY_NUM, calNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Ccopy, CcopyTilingData)
} // namespace optiling
#endif // CCOPY_TILING_H
