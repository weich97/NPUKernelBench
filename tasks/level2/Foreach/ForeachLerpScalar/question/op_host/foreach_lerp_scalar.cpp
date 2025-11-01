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
    class ForeachLerpScalar: public OpDef {
    public:
    explicit ForeachLerpScalar(const char* name) : OpDef(name) {
        std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
        std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
        std::vector<ge::DataType> scalar_dtype_list(tensor_dtype_list.size(), ge::DT_FLOAT);
        this->Input("x1")
        .ParamType(DYNAMIC)
        .DataType(tensor_dtype_list)
        .Format(format_list)
        .UnknownShapeFormat(format_list)
        .AutoContiguous();
        this->Input("x2")
        .ParamType(DYNAMIC)
        .DataType(tensor_dtype_list)
        .Format(format_list)
        .UnknownShapeFormat(format_list)
        .AutoContiguous();
        this->Input("weight")
        .Scalar()
        .ParamType(REQUIRED)
        .DataType(scalar_dtype_list)
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
    
    OP_ADD(ForeachLerpScalar); 
} // namespace ops