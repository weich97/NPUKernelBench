#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
    class ForeachPowScalarAndTensor: public OpDef {
    public:
        explicit ForeachPowScalarAndTensor(const char* name) : OpDef(name) {
            std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16};
            std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
            std::vector<ge::DataType> scalar_dtype_list;
            std::for_each(tensor_dtype_list.cbegin(), tensor_dtype_list.cend(), [&scalar_dtype_list](ge::DataType dtype){scalar_dtype_list.push_back(DtypeTensor2Scalar(dtype));});
            this->Input("scalar")
                .Scalar()
                .ParamType(REQUIRED)
                .DataType(scalar_dtype_list)
                .Format(format_list)
                .UnknownShapeFormat(format_list)
                .AutoContiguous();
            this->Input("x")
                .ParamType(DYNAMIC)
                .DataType(tensor_dtype_list)
                .Format(format_list)
                .UnknownShapeFormat(format_list)
                .AutoContiguous();
            this->Output("y")
                .ParamType(DYNAMIC)
                .DataType(tensor_dtype_list)
                .Format(format_list)
                .UnknownShapeFormat(format_list)
                .AutoContiguous();
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910_93")
                .AddConfig("ascend910b");
        }
    };
    
    OP_ADD(ForeachPowScalarAndTensor);
}  // namespace ops
