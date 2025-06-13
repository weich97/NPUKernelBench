#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
        return ge::GRAPH_SUCCESS;
    }
}


namespace ops {
class Eye : public OpDef {
public:
    explicit Eye(const char* name) : OpDef(name) {
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_rows").Int();
        this->Attr("num_columns").AttrType(OPTIONAL).Int(0);
        this->Attr("batch_shape").AttrType(OPTIONAL).ListInt({});
        this->Attr("dtype").AttrType(OPTIONAL).Int(0);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Eye);
}

