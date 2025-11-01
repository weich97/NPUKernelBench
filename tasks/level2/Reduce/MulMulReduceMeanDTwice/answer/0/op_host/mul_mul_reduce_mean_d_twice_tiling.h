#ifndef MUL_MUL_REDUCE_MEAN_D_TWICE_TILING_H
#define MUL_MUL_REDUCE_MEAN_D_TWICE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MulMulReduceMeanDTwiceTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF(uint64_t, formerNum); 
  TILING_DATA_FIELD_DEF(uint64_t, formerLength); 
  TILING_DATA_FIELD_DEF(uint64_t, tailNum); 
  TILING_DATA_FIELD_DEF(uint64_t, tailLength); 
  TILING_DATA_FIELD_DEF(uint64_t, tileLength); 
  TILING_DATA_FIELD_DEF(uint64_t, shareSize); 
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MulMulReduceMeanDTwice, MulMulReduceMeanDTwiceTilingData)
}
#endif // MUL_MUL_REDUCE_MEAN_D_TWICE_TILING_H