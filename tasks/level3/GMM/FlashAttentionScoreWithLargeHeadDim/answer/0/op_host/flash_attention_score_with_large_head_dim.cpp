#include "register/op_def_registry.h"
#include "tiling_base.h"
using namespace ge;
using namespace AscendC;
using namespace optiling::FA;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FlashAttentionScoreWithLargeHeadDimTiling* basePtr = new FlashAttentionScoreWithLargeHeadDimTiling(context);
    basePtr->DoTiling();
    delete basePtr;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* q_shape = context->GetInputShape(0);
    const gert::Shape* k_shape = context->GetInputShape(1);
    const gert::Shape* v_shape = context->GetInputShape(2);
    gert::Shape* s_m_shape = context->GetOutputShape(0);
    gert::Shape* s_s_shape = context->GetOutputShape(1);
    gert::Shape* o_shape = context->GetOutputShape(2);
    *s_m_shape = gert::Shape({q_shape->GetDim(0), 1, q_shape->GetDim(1),8});
    *s_s_shape = gert::Shape({q_shape->GetDim(0), 1, q_shape->GetDim(1),8});
    *o_shape = *q_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, DT_FLOAT);
    context->SetOutputDataType(0, DT_FLOAT);
    context->SetOutputDataType(0, DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

}


namespace ops {
class FlashAttentionScoreWithLargeHeadDim : public OpDef {
public:
    explicit FlashAttentionScoreWithLargeHeadDim(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("softmax_max")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("softmax_sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);
        this->Attr("head_num").Int();

        this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910_93")
                          .AddConfig("ascend910b");

    }
};

OP_ADD(FlashAttentionScoreWithLargeHeadDim);
}
