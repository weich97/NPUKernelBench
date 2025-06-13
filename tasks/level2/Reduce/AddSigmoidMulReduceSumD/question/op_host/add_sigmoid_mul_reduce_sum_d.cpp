#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    return ge::GRAPH_SUCCESS;
}
}

namespace ops
{
    class AddSigmoidMulReduceSumD : public OpDef
    {
    public:
        explicit AddSigmoidMulReduceSumD(const char *name) : OpDef(name)
        {
            this->Input("add_0_input0")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND});
            this->Input("add_0_input1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND});
            this->Input("mul_0_input1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND});
            this->Input("mult_1_input1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND});
            this->Input("mult_2_input1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND});

            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
        }
    };

    OP_ADD(AddSigmoidMulReduceSumD);
}