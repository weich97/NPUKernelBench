#include <cstring>
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "mse_loss_tiling.h"

namespace optiling {
    const uint32_t BLOCK_SIZE = 32;

    uint32_t GetSizeOfDataType(gert::TilingContext* context)
    {
        uint32_t sizeOfDataType;
        auto dt = context->GetInputDesc(0)->GetDataType();
        if (dt == 1) {
            sizeOfDataType = 2;
        }
        return sizeOfDataType;
    }

    uint32_t CalculateAlignedLength(uint32_t totalLength, uint32_t ALIGN_NUM) {
        if (ALIGN_NUM == 0) {
            return totalLength;
        }
        return (totalLength % ALIGN_NUM != 0) 
            ? ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM
            : totalLength;
    }

    void CalculateTileParameters(uint32_t totalLengthAligned, uint32_t block_dim,
    uint32_t ALIGN_NUM, uint32_t ub_block_num,
    uint32_t& blockLength, uint32_t& tile_num,
    uint32_t& tileLength, uint32_t& lastTileLength) {
    blockLength = (block_dim != 0) ? totalLengthAligned / block_dim : 0;

    if (blockLength == 0) {
        tile_num = 0;
        tileLength = 0;
        lastTileLength = 0;
        return;
    }

    if (ub_block_num != 0 && ALIGN_NUM != 0) {
        tile_num = blockLength / ALIGN_NUM / ub_block_num;

        if (ALIGN_NUM != 0 && ub_block_num != 0 && (((blockLength / ALIGN_NUM) % ub_block_num == 0 || tile_num == 0))) {
            tile_num = (tile_num == 0) ? 1 : tile_num;
            if (blockLength < ub_block_num * ALIGN_NUM) {
                tileLength = (ALIGN_NUM != 0) ? ((blockLength / ALIGN_NUM) + 1) / 2 * 2 * ALIGN_NUM : 0;
            } else {
                tileLength = ub_block_num * ALIGN_NUM;
            }
            lastTileLength = tileLength;
        }
    }else {
        tile_num += 1;
        tileLength = ub_block_num * ALIGN_NUM;
        lastTileLength = blockLength - (tile_num - 1) * tileLength;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    MseLossTilingData tiling;

    uint32_t sizeOfDataType = GetSizeOfDataType(context);
    if (sizeOfDataType == 0) {
    return ge::GRAPH_FAILED;
    }

    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeOfDataType;
    uint32_t ub_block_num = (1024 % 2 != 0) ? 1023 : 1024;

    int64_t reduction = *context->GetAttrs()->GetInt(0);
    tiling.set_mode(reduction);

    uint32_t totalLengthAligned = CalculateAlignedLength(totalLength, ALIGN_NUM);
    tiling.set_totalLength(totalLength);

    context->SetBlockDim(1);
    auto block_dim = context->GetBlockDim();

    uint32_t blockLength, tile_num, tileLength, lastTileLength;
    CalculateTileParameters(totalLengthAligned, block_dim, ALIGN_NUM, 
        ub_block_num, blockLength, tile_num, 
        tileLength, lastTileLength);

    tiling.set_blockLength(blockLength);
    tiling.set_tileNum(tile_num);
    tiling.set_tileLength(tileLength);
    tiling.set_lastTileLength(lastTileLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    std::cout << "*******************START*******************" << std::endl;
    std::cout << "coreNum = " << 1 << std::endl;
    std::cout << "mode = " << tiling.get_mode() << std::endl;
    std::cout << "totalLength = " << tiling.get_totalLength() << std::endl;
    std::cout << "blockLength = " << tiling.get_blockLength() << std::endl;
    std::cout << "tileNum = " << tiling.get_tileNum() << std::endl;
    std::cout << "tileLength = " << tiling.get_tileLength() << std::endl;
    std::cout << "lastTileLength = " << tiling.get_lastTileLength() << std::endl;
    std::cout << "*******************END*******************" << std::endl;
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class MseLoss : public OpDef {
public:
    explicit MseLoss(const char* name) : OpDef(name)
    {
        this->Input("predict")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("label")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("reduction").AttrType(OPTIONAL).Int(1); //mean 1, sum 2, none3
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93")
                      .AddConfig("ascend910b");
    }
};
OP_ADD(MseLoss);
}