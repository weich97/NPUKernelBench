#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class MaskedSelectV3 : public OpDef {
 public:
  explicit MaskedSelectV3(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT16, ge::DT_UINT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT32, ge::DT_UINT64, ge::DT_INT64, ge::DT_DOUBLE})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("mask")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .OutputShapeDependOnCompute()
        .ParamType(REQUIRED)
        .DataType({ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT16, ge::DT_UINT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT32, ge::DT_UINT64, ge::DT_INT64, ge::DT_DOUBLE})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    OpAICoreConfig aicoreConfig;
    aicoreConfig.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(MaskedSelectV3);
}  // namespace ops