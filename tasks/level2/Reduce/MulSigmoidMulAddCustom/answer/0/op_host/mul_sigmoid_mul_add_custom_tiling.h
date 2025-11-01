#ifndef MUL_SIGMOID_MUL_ADD_CUSTOM_TILING_H
#define MUL_SIGMOID_MUL_ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MulSigmoidMulAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLen);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, completeTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, partTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, completeTileLen);
  TILING_DATA_FIELD_DEF(uint32_t, partTileLen);
  TILING_DATA_FIELD_DEF(uint32_t, totalTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, frontBlockNum);
  TILING_DATA_FIELD_DEF(uint32_t, latterBlockNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumInFrontBlock);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumInLatterBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MulSigmoidMulAddCustom, MulSigmoidMulAddCustomTilingData)
}
#endif // MUL_SIGMOID_MUL_ADD_CUSTOM_TILING_H
