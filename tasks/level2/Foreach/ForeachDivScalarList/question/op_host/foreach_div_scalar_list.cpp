#include <algorithm>
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
inline ge::DataType DtypeTensor2Scalar(ge::DataType dtype) {
    switch(dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_FLOAT:
        case ge::DT_BF16:
            return ge::DT_FLOAT;
        case ge::DT_INT32:
            return ge::DT_INT64;
        default:
            return ge::DT_UNDEFINED;
    }
    return ge::DT_UNDEFINED;
}
class ForeachDivScalarList: public OpDef {
public:
    explicit ForeachDivScalarList(const char* name) : OpDef(name) {
        std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
        std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
        std::vector<ge::DataType> scalar_dtype_list;
        std::for_each(tensor_dtype_list.cbegin(), tensor_dtype_list.cend(), [&scalar_dtype_list](ge::DataType dtype){scalar_dtype_list.push_back(DtypeTensor2Scalar(dtype));});
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();
        this->Input("scalars")
            .ScalarList()
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
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(ForeachDivScalarList);
} // namespace ops