#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "padding_matmul_tiling.h"

namespace optiling {
const uint32_t align = 256;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);
    TilingData tiling;
    uint32_t m = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    uint32_t k = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    uint32_t n = context->GetInputShape(1)->GetStorageShape().GetDim(1);
    uint32_t padding_k = (k + align - 1) / align * align;
    uint32_t padding_n = (n + align - 1) / align * align;

    tiling.set_m(m);
    tiling.set_k(k);
    tiling.set_n(n);
    tiling.set_padding_k(padding_k);
    tiling.set_padding_n(padding_n);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    auto dataType = context->GetInputDesc(0)->GetDataType();
    size_t lenWorkspace = static_cast<size_t>(m * padding_k + k * padding_n) * GetSizeByDataType(dataType);
    currentWorkspace[0] = lenWorkspace + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class PaddingMatmul : public OpDef {
public:
    explicit PaddingMatmul(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};
OP_ADD(PaddingMatmul);
} // namespace ops