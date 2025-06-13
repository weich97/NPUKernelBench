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
    class ApplyAdamWV2 : public OpDef {
    public:
        explicit ApplyAdamWV2(const char* name) : OpDef(name)
        {
            this->Input("var")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("m")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("v")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("grad")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("step")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("max_grad_norm")
                    .ParamType(OPTIONAL)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("lr")
                    .AttrType(OPTIONAL)
                    .Float(0.1);
            this->Attr("beta1")
                    .AttrType(OPTIONAL)
                    .Float(0.1);
            this->Attr("beta2")
                    .AttrType(OPTIONAL)
                    .Float(0.1);
            this->Attr("weight_decay")
                    .AttrType(OPTIONAL)
                    .Float(0.1);
            this->Attr("eps")
                    .AttrType(OPTIONAL)
                    .Float(0.1);
            this->Attr("amsgrad")
                    .AttrType(OPTIONAL)
                    .Bool(false);
            this->Attr("maximize")
                    .AttrType(OPTIONAL)
                    .Bool(false);
            OpAICoreConfig aicore_config;
            aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "apply_adam_wv2")
            .ExtendCfgInfo("opInterface.value", "apply_adam_wv2");
            this->AICore().AddConfig("ascend910b", aicore_config);
            this->AICore().AddConfig("ascend910_93", aicore_config);
            this->AICore().SetTiling(optiling::TilingFunc);
        }
    };

    OP_ADD(ApplyAdamWV2);

}  // namespace ops

