#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "group_gemm_tiling.h"

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

template <class T>
size_t GetWorkspaceLen(T m, T n, T align)
{
    return static_cast<size_t>(m) * RoundUp(n, align);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);

    auto attrs = context->GetAttrs();
    auto mList = attrs->GetListInt(0)->GetData();
    auto kList = attrs->GetListInt(1)->GetData();
    auto nList = attrs->GetListInt(2)->GetData();

    size_t sizeWA = 0;
    size_t sizeWB = 0;
    size_t sizeW = 0;

    TilingData tiling;
    auto groupCount = attrs->GetListInt(0)->GetSize();
    auto dataType = context->GetInputDesc(0)->GetDataType();
    size_t dataTypeByte = GetSizeByDataType(dataType);
    uint32_t align = alignByByte / dataTypeByte;

    int64_t mListArr[MAX_TENSOR_COUNT];
    int64_t kListArr[MAX_TENSOR_COUNT];
    int64_t nListArr[MAX_TENSOR_COUNT];

    for (uint32_t i = 0; i < groupCount; ++i) {
        mListArr[i] = mList[i];
        kListArr[i] = kList[i];
        nListArr[i] = nList[i];

        sizeWA += GetWorkspaceLen(mListArr[i], kListArr[i], static_cast<int64_t>(align));
        sizeWB += GetWorkspaceLen(kListArr[i], nListArr[i], static_cast<int64_t>(align));
        sizeW += static_cast<size_t>(mListArr[i] * nListArr[i]);
    }

    sizeWA *= dataTypeByte;
    sizeWB *= dataTypeByte;
    sizeW *= dataTypeByte;

    tiling.set_groupCount(groupCount);
    tiling.set_mList(mListArr);
    tiling.set_kList(kListArr);
    tiling.set_nList(nListArr);
    tiling.set_align(align);
    tiling.set_paddingASize(sizeWA);
    tiling.set_paddingBSize(sizeWB);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t lenWorkspace = sizeWA + sizeWB + sizeW;
    currentWorkspace[0] = lenWorkspace + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class GroupGemm : public OpDef {
public:
    explicit GroupGemm(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("alpha")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("mList")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("kList")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("nList")
            .AttrType(REQUIRED)
            .ListInt();

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};
OP_ADD(GroupGemm);
} // namespace ops