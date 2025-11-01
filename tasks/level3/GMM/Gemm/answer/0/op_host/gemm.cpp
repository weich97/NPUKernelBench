#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "gemm_tiling.h"

namespace optiling {
constexpr uint32_t alignByByte = 512;

bool IsNeedPadding(uint32_t n, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (n < 65536) {
        return n % align != 0;
    } else {
        return true;
    }
}

template <class T>
constexpr T RoundUp(const T &val, const T align)
{
    return (val + align - 1) / align * align;
}

size_t GetWorkspaceLen(uint32_t m, uint32_t n, uint32_t align)
{
    return static_cast<size_t>(m) * RoundUp(n, align);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);
    TilingData tiling;
    uint32_t m = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    uint32_t k = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    uint32_t n = context->GetInputShape(1)->GetStorageShape().GetDim(1);
    auto dataType = context->GetInputDesc(0)->GetDataType();
    size_t dataTypeByte = GetSizeByDataType(dataType);
    uint32_t align = alignByByte / dataTypeByte;
    auto* attrs = context->GetAttrs();
    const float* alpha = attrs->GetAttrPointer<float>(0);
    const float* beta = attrs->GetAttrPointer<float>(1);

    bool isNeedPaddingA = IsNeedPadding(k, align);
    bool isNeedPaddingB = IsNeedPadding(n, align);
    size_t sizeWA = 0;
    size_t sizeWB = 0;
    if (isNeedPaddingA) {
        sizeWA = GetWorkspaceLen(m, k, align) * dataTypeByte;
    }
    if (isNeedPaddingB) {
        sizeWB = GetWorkspaceLen(k, n, align) * dataTypeByte;
    }

    tiling.set_m(m);
    tiling.set_k(k);
    tiling.set_n(n);
    tiling.set_align(align);
    tiling.set_paddingASize(sizeWA);
    tiling.set_paddingBSize(sizeWB);
    tiling.set_alpha(static_cast<float>(*alpha));
    tiling.set_beta(static_cast<float>(*beta));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t lenWorkspace = sizeWA + sizeWB + static_cast<size_t>(m) * n * dataTypeByte;
    currentWorkspace[0] = lenWorkspace + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class Gemm : public OpDef {
public:
    explicit Gemm(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("alpha")
            .AttrType(REQUIRED)
            .Float();
        this->Attr("beta")
            .AttrType(REQUIRED)
            .Float();

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};
OP_ADD(Gemm);
} // namespace ops