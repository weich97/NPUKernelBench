#ifndef UNALIGN_ADD_TILING_H
#define UNALIGN_ADD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UnalignAdd, TilingData)
} // namespace optiling
#endif // UNALIGN_ADD_TILING_H