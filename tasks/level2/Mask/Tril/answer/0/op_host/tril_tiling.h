#ifndef TRIL_TILING_H
#define TRIL_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
  BEGIN_TILING_DATA_DEF(TrilTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLengthAligned);
  TILING_DATA_FIELD_DEF(int32_t, matrixNum);
  TILING_DATA_FIELD_DEF(int32_t, matrixSize);
  TILING_DATA_FIELD_DEF(int32_t, rowLength);
  TILING_DATA_FIELD_DEF(int32_t, columnLength);
  TILING_DATA_FIELD_DEF(int32_t, diagVal);
  TILING_DATA_FIELD_DEF(int32_t, loopCnt);
  TILING_DATA_FIELD_DEF(uint32_t, fullTileLength);
  TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
  TILING_DATA_FIELD_DEF(int32_t, fullCnt);
  TILING_DATA_FIELD_DEF(int32_t, lastCnt);
  TILING_DATA_FIELD_DEF(uint32_t, alignNum);
  TILING_DATA_FIELD_DEF(uint32_t, typeSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Tril, TrilTilingData)
}
#endif // TRIL_TILING_H