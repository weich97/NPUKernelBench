#include "select_reduce_max_d_sub_exp_reduce_sum_d_real_div_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SelectReduceMaxDSubExpReduceSumDRealDivTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    uint32_t totalLength = x1_shape->GetStorageShape().GetShapeSize();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t dimNum = x1_shape->GetStorageShape().GetDimNum();
    uint32_t rowDim = x1_shape->GetStorageShape().GetDim(0);
    uint32_t srcLastDim = x1_shape->GetStorageShape().GetDim(dimNum - 1);
    uint32_t tileNum = 2;
    if (rowDim <= aivNum) {
        aivNum = rowDim;
    }
    else {
        aivNum = rowDim / ((rowDim-1) / aivNum + 1);
    }
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(tileNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(aivNum);

    std::cout << "*******************START*******************" << std::endl;
    std::cout << "coreNum = " << aivNum << std::endl;
    std::cout << "totalLength = " << tiling.get_totalLength() << std::endl;
    std::cout << "tileNum = " << tiling.get_tileNum() << std::endl;
    std::cout << "tmpSize = " << tiling.get_tmpSize() << std::endl;
    std::cout << "*******************END*******************" << std::endl;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    auto dtype = context->GetInputDataType(1);
    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SelectReduceMaxDSubExpReduceSumDRealDiv : public OpDef {
public:
    explicit SelectReduceMaxDSubExpReduceSumDRealDiv(const char* name) : OpDef(name)
    {
        this->Input("sel")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("input1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("input2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("ouput")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SelectReduceMaxDSubExpReduceSumDRealDiv);
}