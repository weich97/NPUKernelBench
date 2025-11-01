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

static const int64_t AXIS_DEFAULT = -2;

static const std::vector<ge::DataType> varDataType910b = {
    ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32,
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
    ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32,
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT
};

static const std::vector<ge::DataType> indiceDataType910b = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64
};

static const std::vector<ge::DataType> maskDataType910b = {
    ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
    ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
    ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
    ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8
};

static const std::vector<ge::Format> format910b = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};

class ScatterList : public OpDef {
public:
    explicit ScatterList(const char* name) : OpDef(name) {
        this->Input("var")
            .ParamType(DYNAMIC)
            .DataType(varDataType910b)
            .Format(format910b)
            .UnknownShapeFormat(format910b);
        this->Input("indice")
            .ParamType(REQUIRED)
            .DataType(indiceDataType910b)
            .Format(format910b)
            .UnknownShapeFormat(format910b);
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType(varDataType910b)
            .Format(format910b)
            .UnknownShapeFormat(format910b);
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType(maskDataType910b)
            .Format(format910b)
            .UnknownShapeFormat(format910b);
        this->Output("var")
            .ParamType(DYNAMIC)
            .DataType(varDataType910b)
            .Format(format910b)
            .UnknownShapeFormat(format910b);
        this->Attr("reduce")
            .AttrType(OPTIONAL)
            .String("update");
        this->Attr("axis")
            .AttrType(OPTIONAL)
            .Int(AXIS_DEFAULT);

        this->AICore()
        .SetTiling(optiling::TilingFunc)
        .AddConfig("ascend910_93")
        .AddConfig("ascend910b");
    }
};

OP_ADD(ScatterList);
}  // namespace ops