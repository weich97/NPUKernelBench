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
static const std::vector<ge::DataType> g_xDtypes = {
    ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT8,      ge::DT_UINT8,   ge::DT_INT16, ge::DT_UINT16,   ge::DT_INT32,
    ge::DT_INT64,   ge::DT_BOOL,   ge::DT_COMPLEX64, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8,     ge::DT_UINT8,
    ge::DT_INT16,   ge::DT_UINT16, ge::DT_INT32,     ge::DT_INT64,   ge::DT_BOOL,  ge::DT_COMPLEX64};

static const std::vector<ge::DataType> g_yDtypes = g_xDtypes;

static const std::vector<ge::DataType> g_seqLengthsDtypes = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::Format> g_formats = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    
class ReverseSequence : public OpDef {
public:
    explicit ReverseSequence(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType(g_xDtypes).Format(g_formats);
        this->Input("seq_lengths").ParamType(REQUIRED).DataType(g_seqLengthsDtypes).Format(g_formats);
        this->Output("y").ParamType(REQUIRED).DataType(g_yDtypes).Format(g_formats);
        this->Attr("seq_dim").Int();
        this->Attr("batch_dim").AttrType(OPTIONAL).Int(0);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().SetTiling(optiling::TilingFunc);
    }
};

OP_ADD(ReverseSequence);
}  // namespace ops