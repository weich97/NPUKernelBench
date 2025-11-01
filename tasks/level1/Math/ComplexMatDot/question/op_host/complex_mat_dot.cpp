#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
class ComplexMatDot : public OpDef {
public:
    explicit ComplexMatDot(const char *name) : OpDef(name)
    {
        this->Input("matx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND});
        this->Input("maty")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND});
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND});
        this->Attr("m").AttrType(OPTIONAL).Int(1);
        this->Attr("n").AttrType(OPTIONAL).Int(1);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(ComplexMatDot);
} // namespace ops
