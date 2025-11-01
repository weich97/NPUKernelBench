#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "matmul_bias_tiling.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);
    TilingData tiling;
    uint32_t m = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    uint32_t k = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    uint32_t n = context->GetInputShape(1)->GetStorageShape().GetDim(1);

    tiling.set_m(m);
    tiling.set_k(k);
    tiling.set_n(n);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class MatmulBias : public OpDef {
public:
    explicit MatmulBias(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};
OP_ADD(MatmulBias);
} // namespace ops