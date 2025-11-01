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

static const std::vector<ge::DataType> inputDataType = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, ge::DT_INT64, ge::DT_DOUBLE, ge::DT_INT8};

static const std::vector<ge::DataType> idxDataType = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::Format> format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class StridedSliceAssignV2 : public OpDef {
public:
    explicit StridedSliceAssignV2(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("input_value")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("begin")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("end")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("strides")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("axes")
            .ParamType(OPTIONAL)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        this->AICore().SetTiling(optiling::TilingFunc);

    }
};
OP_ADD(StridedSliceAssignV2);
}  // namespace ops