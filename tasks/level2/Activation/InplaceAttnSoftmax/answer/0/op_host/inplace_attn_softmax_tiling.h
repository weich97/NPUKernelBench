#ifndef INPLACE_ATTN_SOFTMAX_TILING_H
#define INPLACE_ATTN_SOFTMAX_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InplaceAttnSoftmaxTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rowLen);               // 行数
  TILING_DATA_FIELD_DEF(uint32_t, colLen);               // 列数, 输入x的一半
  TILING_DATA_FIELD_DEF(uint32_t, rowLenPerHeadCore);    // 头核处理行数
  TILING_DATA_FIELD_DEF(uint32_t, rowLenPerTailCore);    // 尾核处理行数
  TILING_DATA_FIELD_DEF(uint32_t, basicRowLenHeadCore);  // 头核每次计算的行数 类似于TILE_LENGTH
  TILING_DATA_FIELD_DEF(uint32_t, basicRowLenTailCore);  // 尾核每次计算的行数
  TILING_DATA_FIELD_DEF(uint32_t, basicColLen);          // 每次计算的列数
  TILING_DATA_FIELD_DEF(uint32_t, headCoreNum);          // 使用的head核数
  TILING_DATA_FIELD_DEF(uint32_t, realCoreNum);          // 实际使用的核数
  TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InplaceAttnSoftmax, InplaceAttnSoftmaxTilingData)

struct InplaceAttnSoftmaxCompileInfo {
    uint32_t totalCore = 1;
    uint32_t ubSize = 0;
    uint32_t inputDataByte = 2;
    uint32_t dataNumSingleUb = 1;  // UB空间可处理的最大数据量
    uint32_t block_num = 1;        // 32B对齐使用
    
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
