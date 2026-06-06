#ifndef INPLACE_ATTN_SOFTMAX_TILING_H
#define INPLACE_ATTN_SOFTMAX_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InplaceAttnSoftmaxTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rowLen); // Implementation note.
  TILING_DATA_FIELD_DEF(uint32_t, colLen); // Implementation note.
  TILING_DATA_FIELD_DEF(uint32_t, rowLenPerHeadCore); // Implementation note.
  TILING_DATA_FIELD_DEF(uint32_t, rowLenPerTailCore); // Implementation note.
  TILING_DATA_FIELD_DEF(uint32_t, basicRowLenHeadCore); // Implementation note.
  TILING_DATA_FIELD_DEF(uint32_t, basicRowLenTailCore); // Implementation note.
  TILING_DATA_FIELD_DEF(uint32_t, basicColLen); // Implementation note.
  TILING_DATA_FIELD_DEF(uint32_t, headCoreNum); // Implementation note.
  TILING_DATA_FIELD_DEF(uint32_t, realCoreNum); // Implementation note.
  TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InplaceAttnSoftmax, InplaceAttnSoftmaxTilingData)

struct InplaceAttnSoftmaxCompileInfo {
    uint32_t totalCore = 1;
    uint32_t ubSize = 0;
    uint32_t inputDataByte = 2;
    uint32_t dataNumSingleUb = 1; // Implementation note.
    uint32_t block_num = 1; // Implementation note.
    
};                                

struct InplaceAttnSoftmaxTilingParam {
    uint32_t optBaseRowLenHeadCore = 1;
    uint32_t optBaseRowLenTailCore = 1;
    uint32_t optBaseColLen = 1;
    uint32_t rowLenPerHeadCore = 0;
    uint32_t rowLenPerTailCore = 0;
    uint32_t headCoreNum = 0;
    uint32_t coreNumUsed = 0;
};

enum class InplaceAttnSoftmaxTilingKey : int32_t {
  TILINGKEY_FP16 = 101,
  TILINGKEY_FP16_BIGSHAPE = 111,
  TILINGKEY_BF16 = 201,
  TILINGKEY_BF16_BIGSHAPE = 211,
  TILINGKEY_FP32 = 301,
  TILINGKEY_FP32_BIGSHAPE = 311
};
}
#endif // INPLACE_ATTN_SOFTMAX_TILING_H
