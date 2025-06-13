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
class FeedsRepeat : public OpDef {
 public:
  explicit FeedsRepeat(const char* name) : OpDef(name) {
    this->Input("feeds")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("feeds_repeat_times")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("output_feeds_size").AttrType(REQUIRED).Int();
    
    this->AICore()
        .SetTiling(optiling::TilingFunc)
        .AddConfig("ascend910_93")
        .AddConfig("ascend910b");
  }
};
OP_ADD(FeedsRepeat);
}  // namespace ops