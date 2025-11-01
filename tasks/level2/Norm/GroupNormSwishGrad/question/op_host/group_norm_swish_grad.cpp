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

class GroupNormSwishGrad : public OpDef {
public:
    explicit GroupNormSwishGrad(const char* name) : OpDef(name)
    {
        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dgamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dbeta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("num_groups")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("data_format")
            .AttrType(OPTIONAL)
            .String("NCHW");
        this->Attr("swish_scale")
            .AttrType(OPTIONAL)
            .Float(1.0);
        this->Attr("dgamma_is_require")
            .AttrType(OPTIONAL)
            .Bool(true);
        this->Attr("dbeta_is_require")
            .AttrType(OPTIONAL)
            .Bool(true);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};

OP_ADD(GroupNormSwishGrad);
}  // namespace ops
