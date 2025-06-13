#include "gelu_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

namespace optiling
{
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t BUFFER_NUM = 2;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) // Tiling函数实现
    {
        GeluGradTilingData tiling;
        uint64_t ubSize;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        auto coreNum = ascendcPlatform.GetCoreNum();
        auto socVersion = ascendcPlatform.GetSocVersion();
        if (socVersion != platform_ascendc::SocVersion::ASCEND910B && socVersion != platform_ascendc::SocVersion::ASCEND310B && context->GetInputDesc(0)->GetDataType() == ge::DT_BF16)
        {
            return ge::GRAPH_FAILED;
        }

        uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
        uint32_t inputLength = inputNum * typeLength;
        uint32_t inputBytes = inputLength / inputNum;

        uint32_t ubDataNumber = 10;
        uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
        uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

        uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        coreNum = (coreNum < inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
        coreNum = (coreNum >= 1) ? coreNum : 1;
        uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
        uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;

        uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
        uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
        uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
        uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
        smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

        everyCoreInputBlockNum += 1;
        uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
        uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
        uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
        uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
        bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

        uint32_t versionNum = 0;
        if (socVersion == platform_ascendc::SocVersion::ASCEND310B)
        {
            versionNum = 1;
        }

        tiling.set_smallCoreDataNum(smallCoreDataNum);
        // 一个小核数据个数
        tiling.set_bigCoreDataNum(bigCoreDataNum);
        // 一个大核数据个数
        tiling.set_tileDataNum(tileDataNum);
        // 一次搬运的数据个数
        tiling.set_smallTailDataNum(smallTailDataNum);
        // 小核尾块数据个数
        tiling.set_bigTailDataNum(bigTailDataNum);
        // 大核尾块数据个数
        tiling.set_finalSmallTileNum(finalSmallTileNum);
        // 小核搬运次数
        tiling.set_finalBigTileNum(finalBigTileNum);
        // 大核搬运次数
        tiling.set_tailBlockNum(tailBlockNum);
        // 大核数
        tiling.set_versionNum(versionNum);

        context->SetBlockDim(coreNum);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        std::cout << "*******************START*******************" << std::endl;
        std::cout << "coreNum = " << coreNum << std::endl;
        std::cout << "smallCoreDataNum = " << tiling.get_smallCoreDataNum() << std::endl;
        std::cout << "bigCoreDataNum = " << tiling.get_bigCoreDataNum() << std::endl;
        std::cout << "finalBigTileNum = " << tiling.get_finalBigTileNum() << std::endl;
        std::cout << "finalSmallTileNum = " << tiling.get_finalSmallTileNum() << std::endl;
        std::cout << "tileDataNum = " << tiling.get_tileDataNum() << std::endl;
        std::cout << "smallTailDataNum = " << tiling.get_smallTailDataNum() << std::endl;
        std::cout << "bigTailDataNum = " << tiling.get_bigTailDataNum() << std::endl;
        std::cout << "tailBlockNum = " << tiling.get_tailBlockNum() << std::endl;
        std::cout << "versionNum = " << tiling.get_versionNum() << std::endl;
        std::cout << "*******************END*******************" << std::endl;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops
{
    class GeluGrad : public OpDef
    {
    public:
        explicit GeluGrad(const char *name) : OpDef(name)
        {
            this->Input("dy")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("z")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
        }
    };

    OP_ADD(GeluGrad);
}
