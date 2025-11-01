#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "splitk_matmul_tiling.h"

namespace optiling {
template <class T>
constexpr T CeilDiv(const T dividend, const T divisor)
{
    return (dividend + divisor - 1) / divisor;
}

// A splitk strategy, which may not be the optimal one.
uint32_t GetSplitkFactor(uint32_t m, uint32_t n, uint32_t k, uint32_t m0, uint32_t n0, uint32_t k0, uint32_t aicCoreNum)
{
    uint32_t maxSplitkFactor;
    if (k <= 1024) {
        // When k is less than or equal to 1024, it can be divided into at most 2 parts.
        maxSplitkFactor = 2;
    } else if (k <= 2048) {
        // When k is less than or equal to 2048, it can be divided into at most 4 parts.
        maxSplitkFactor = 4;
    } else if (k <= 4096) {
        // When k is less than or equal to 4096, it can be divided into at most 8 parts.
        maxSplitkFactor = 8;
    } else {
        // else it can be divided into at most 16 parts.
        maxSplitkFactor = 16;
    }
    uint32_t splitkFactor = 1;
    uint32_t baseTilesCount = CeilDiv(m, m0) * CeilDiv(n, n0);
    splitkFactor = std::min(aicCoreNum / baseTilesCount, maxSplitkFactor);
    // Prevent the split factor form being less than 1
    splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(1));
    if (baseTilesCount < aicCoreNum) {
        while (splitkFactor + 1 <= maxSplitkFactor &&
            CeilDiv(baseTilesCount * splitkFactor, aicCoreNum) >= CeilDiv(baseTilesCount, aicCoreNum) * splitkFactor) {
            splitkFactor += 1;
        }
    }
    // Ensure that splitkFactor is less than the number of base tiels in the k direction.
    splitkFactor = std::min(CeilDiv(k, k0), splitkFactor);
    // If k is very large, splitting k can lead to better cache utilization.
    // If k is greater than 8192.
    if (k > 8192) {
        // split the k direction into at least 2 parts.
        splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(2));
    }
    // If k is greater than 32768.
    if (k > 32768) {
        // split the k direction into at least 4 parts.
        splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(4));
    }
    return splitkFactor;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);
    TilingData tiling;
    uint32_t m = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    uint32_t k = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    uint32_t n = context->GetInputShape(1)->GetStorageShape().GetDim(1);
    uint32_t splitkFactor = GetSplitkFactor(m, n, k, 128, 256, 256, aicCoreNum);

    tiling.set_m(m);
    tiling.set_k(k);
    tiling.set_n(n);
    tiling.set_splitkFactor(splitkFactor);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t lenWorkspace = static_cast<size_t>(m * n * splitkFactor) * sizeof(float);
    currentWorkspace[0] = lenWorkspace + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class SplitkMatmul : public OpDef {
public:
    explicit SplitkMatmul(const char *name) : OpDef(name)
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
OP_ADD(SplitkMatmul);
} // namespace ops