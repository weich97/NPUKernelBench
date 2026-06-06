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
        // Implementation note.
        std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
        std::vector<ge::Format>   format_list(tensor_dtype_list.size(), ge::FORMAT_ND);

        // Implementation note.
        std::vector<ge::DataType> scalar_tensor_dtype_list = {
            DtypeScalarToTensor2(ge::DT_FLOAT16),
            DtypeScalarToTensor2(ge::DT_FLOAT),
            DtypeScalarToTensor2(ge::DT_BF16)
        };

        // Implementation note.
        // Implementation note.
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();

        // Implementation note.
        this->Input("roundMode")
            .ParamType(REQUIRED)
            .DataType(scalar_tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list);

        // Implementation note.
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();

        // Implementation note.
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};
OP_ADD(ForeachRoundOffNumber);
}  // namespace ops
