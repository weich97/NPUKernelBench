#include "gelu_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

const uint32_t BUFFER_NUM = 2;
const uint32_t BLOCK_SIZE = 32;
const int32_t DATATYPE1 = 2;
const int32_t DATATYPE2 = 4;
const int32_t BYTESIZE = 8;
namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        GeluTilingData tiling;
        int32_t NUM = 2;
        int32_t DATATYPE1 = 2;
        int32_t DATATYPE2 = 4;

        uint32_t sizeofdatatype;
        uint32_t totalLengthAligned;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint64_t ub_size;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        auto aivNum = ascendcPlatform.GetCoreNum();

        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        auto dt = context->GetInputTensor(0)->GetDataType();
        if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16)
        {
            sizeofdatatype = DATATYPE1;
            NUM = 4;    //需要额外分出一半空间给TBuffer
        }
        else
        {
            sizeofdatatype = DATATYPE2;
        }

        uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
        uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
        tiling_size = tiling_size <= BYTESIZE ? tiling_size : tiling_size / BYTESIZE * BYTESIZE;

        uint32_t block_size = tiling_size * ALIGN_NUM;
        aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);
	    aivNum = aivNum >= 1 ? aivNum : 1;

        uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * BYTESIZE) * (ALIGN_NUM * BYTESIZE);
        uint32_t core_remain = totalLength - aivNum * core_size;
        tiling.set_totalLength(totalLength);
        tiling.set_ALIGN_NUM(ALIGN_NUM);
        tiling.set_tiling_size(tiling_size);
        tiling.set_block_size(block_size);
        tiling.set_aivNum(aivNum);
        tiling.set_core_size(core_size);
        tiling.set_core_remain(core_remain);

        context->SetBlockDim(aivNum);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        std::cout << "*******************START*******************" << std::endl;
        std::cout << "coreNum = " << aivNum << std::endl;
        std::cout << "totalLength = " << tiling.get_totalLength() << std::endl;
        std::cout << "tileNum = " << tiling.get_tileNum() << std::endl;
        std::cout << "ALIGN_NUM = " << tiling.get_ALIGN_NUM() << std::endl;
        std::cout << "tiling_size = " << tiling.get_tiling_size() << std::endl;
        std::cout << "block_size = " << tiling.get_block_size() << std::endl;
        std::cout << "aivNum = " << tiling.get_aivNum() << std::endl;
        std::cout << "core_size = " << tiling.get_core_size() << std::endl;
        std::cout << "core_remain = " << tiling.get_core_remain() << std::endl;
        std::cout << "*******************END*******************" << std::endl;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops
{
    class Gelu : public OpDef
    {
    public:
        explicit Gelu(const char *name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910_93")
                          .AddConfig("ascend910b");
        }
    };

    OP_ADD(Gelu);
}

