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
class MulSigmoid : public OpDef {
public:
  explicit MulSigmoid(const char* name) : OpDef(name) {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});

    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});

    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});

    this->Attr("t1").AttrType(REQUIRED).Float(0);
    this->Attr("t2").AttrType(REQUIRED).Float(0);
    this->Attr("t3").AttrType(REQUIRED).Float(0);
    
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(MulSigmoid);
}  // namespace ops



