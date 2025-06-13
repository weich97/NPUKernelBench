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

class AddLayerNorm : public OpDef {
#define ALL_FORMAT_ND_910                                                                                        \
    {                                                                                                            \
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, \
            ge::FORMAT_ND, ge::FORMAT_ND                                                                         \
    }
#define ALL_FORMAT_ND_310            \
    {                                \
        ge::FORMAT_ND, ge::FORMAT_ND \
    }

public:
    explicit AddLayerNorm(const char *name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Output("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT16,
                ge::DT_BF16,
                ge::DT_FLOAT})
            .Format(ALL_FORMAT_ND_910)
            .UnknownShapeFormat(ALL_FORMAT_ND_910);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5);
        this->Attr("additional_output").AttrType(OPTIONAL).Bool(false);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().SetTiling(optiling::TilingFunc);
    }
};

OP_ADD(AddLayerNorm);
}  // namespace ops
