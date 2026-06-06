#include "mul_sigmoid_mul_add_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static int64_t CeilDivision(int64_t num1, int64_t num2) {
  if (num2 == 0) {
    return 0;
  }
  return (num1 + num2 - 1) / num2;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MulSigmoidMulAddCustomTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    // Implementation note.
    uint32_t totalLen = x1_shape->GetStorageShape().GetShapeSize();
    uint32_t maxTileLen = 4096;

    // Implementation note.
    uint32_t totalTileNum = CeilDivision(totalLen, maxTileLen);

    // Implementation note.
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    auto blockDim = (totalTileNum < aivNum) ? totalTileNum : aivNum;

    // Implementation note.
    uint32_t completeTileNum, partTileNum;
    uint32_t completeTileLen, partTileLen;

    completeTileNum= totalLen / maxTileLen;     
    completeTileLen  = maxTileLen;  
    if (0 == (totalLen % maxTileLen)) {
        // Implementation note.
        partTileNum = 0;
        partTileLen = 0;
    } else {
        // Implementation note.
        partTileNum = 1;
        partTileLen = totalLen % maxTileLen;
    }

    // Implementation note.
    uint32_t frontBlockNum, latterBlockNum;
    uint32_t tileNumInFrontBlock, tileNumInLatterBlock;
    if (0 == (totalTileNum % blockDim)) {
        frontBlockNum = blockDim;
        tileNumInFrontBlock = totalTileNum / blockDim;

        latterBlockNum = 0;
        tileNumInLatterBlock = 0;
    } else {
        tileNumInFrontBlock = totalTileNum / blockDim + 1; 
        frontBlockNum = totalTileNum % blockDim;
                
        tileNumInLatterBlock = totalTileNum / blockDim;
        latterBlockNum = blockDim - frontBlockNum; 
    }

    context->SetBlockDim(blockDim);

    // Implementation note.
    tiling.set_totalLen(totalLen); 
    tiling.set_blockDim(blockDim); 
    tiling.set_completeTileNum(completeTileNum);
    tiling.set_partTileNum(partTileNum);
    tiling.set_completeTileLen(completeTileLen);
    tiling.set_partTileLen(partTileLen);
    tiling.set_totalTileNum(totalTileNum);
    tiling.set_frontBlockNum(frontBlockNum);
    tiling.set_latterBlockNum(latterBlockNum);
    tiling.set_tileNumInFrontBlock(tileNumInFrontBlock);
    tiling.set_tileNumInLatterBlock(tileNumInLatterBlock);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;

    y_shape->SetDim(0, x1_shape->GetDim(1) * 1);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    auto dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}
}


namespace ops {
class MulSigmoidMulAddCustom : public OpDef {
public:
    explicit MulSigmoidMulAddCustom(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mulscalar1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mulscalar2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("addscalar3")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("outputs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(MulSigmoidMulAddCustom);
}