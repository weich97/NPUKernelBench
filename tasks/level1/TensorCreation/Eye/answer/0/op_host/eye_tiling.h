#ifndef EYE_TILING_H
#define EYE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EyeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, typeKey);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, lasttileLength);
    TILING_DATA_FIELD_DEF(int32_t, num_rows);
    TILING_DATA_FIELD_DEF(int32_t, num_columns);
    TILING_DATA_FIELD_DEF(int32_t, dtype);
    TILING_DATA_FIELD_DEF(int32_t, mark);
    TILING_DATA_FIELD_DEF(int32_t, batchNum);
    TILING_DATA_FIELD_DEF(int32_t, batchSize);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Eye, EyeTilingData)
}
#endif // EYE_TILING_H
