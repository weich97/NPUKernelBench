#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {

inline ge::DataType DtypeScalarToTensor2(ge::DataType dtype) {
    switch (dtype) {
        case ge::DT_FLOAT16: return ge::DT_FLOAT16;
        case ge::DT_FLOAT:   return ge::DT_FLOAT;
        case ge::DT_BF16:    return ge::DT_FLOAT;
        case ge::DT_INT32:   return ge::DT_INT32;
        default:             return ge::DT_UNDEFINED;
    }
}    

class ForeachRoundOffNumber : public OpDef {
public:
    explicit ForeachRoundOffNumber(const char* name) : OpDef(name) {
        /* 1. 准备张量 dtype 与格式 */
        std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
        std::vector<ge::Format>   format_list(tensor_dtype_list.size(), ge::FORMAT_ND);

        /* 2. 准备 scalar dtype（根据 tensor dtype 映射） */
        std::vector<ge::DataType> scalar_tensor_dtype_list = {
            DtypeScalarToTensor2(ge::DT_FLOAT16),
            DtypeScalarToTensor2(ge::DT_FLOAT),
            DtypeScalarToTensor2(ge::DT_BF16)
        };

        /* 3. 注册参数 */
        /*   3.1 输入张量列表 x */
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();

        /*   3.2 输入标量   roundMode */
        this->Input("roundMode")
            .ParamType(REQUIRED)
            .DataType(scalar_tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list);

        /* 3.3 输出张量列表 y */
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();

        /* 4. 核心设备配置 */
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};
OP_ADD(ForeachRoundOffNumber);
}  // namespace ops
