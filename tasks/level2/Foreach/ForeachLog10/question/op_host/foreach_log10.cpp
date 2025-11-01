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
class ForeachLog10: public OpDef {
    public:
        explicit ForeachLog10(const char* name) : OpDef(name) {
            std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
            std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
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
            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
        }
};
OP_ADD(ForeachLog10);
}  // namespace ops
