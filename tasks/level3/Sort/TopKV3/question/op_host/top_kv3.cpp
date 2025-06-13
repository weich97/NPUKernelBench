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
class TopKV3 : public OpDef {
public:
  explicit TopKV3(const char* name) : OpDef(name)
  {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("k")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("values")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Attr("sorted").AttrType(OPTIONAL).Bool(true);
    this->Attr("dim").AttrType(OPTIONAL).Int(-1);
    this->Attr("largest").AttrType(OPTIONAL).Bool(true);

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .ExtendCfgInfo("opFile.value", "top_kv3")
        .ExtendCfgInfo("opInterface.value", "top_kv3")
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

    this->AICore()
        .SetTiling(optiling::TilingFunc)
        .AddConfig("ascend910_93")
        .AddConfig("ascend910b");
  }
};
OP_ADD(TopKV3);
}  // namespace ops