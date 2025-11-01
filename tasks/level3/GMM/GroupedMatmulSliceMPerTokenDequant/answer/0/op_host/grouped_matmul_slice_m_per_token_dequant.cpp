#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "grouped_matmul_slice_m_per_token_dequant_tiling.h"

namespace optiling {
constexpr uint32_t workspaceStages = 2;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);
    TilingData tiling;
    uint32_t m = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    uint32_t k = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    uint32_t n = context->GetInputShape(1)->GetStorageShape().GetDim(2);
    uint32_t groupCount = context->GetInputShape(4)->GetStorageShape().GetDim(0);

    tiling.set_m(m);
    tiling.set_k(k);
    tiling.set_n(n);
    tiling.set_groupCount(groupCount);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t lenWorkspace = static_cast<size_t>(128) * 256 * aicCoreNum * workspaceStages * sizeof(int32_t);
    currentWorkspace[0] = lenWorkspace + sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class GroupedMatmulSliceMPerTokenDequant : public OpDef {
public:
    explicit GroupedMatmulSliceMPerTokenDequant(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("per_token_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("groupList")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};
OP_ADD(GroupedMatmulSliceMPerTokenDequant);
} // namespace ops