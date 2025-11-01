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
class Spence : public OpDef {  
public:  
    explicit Spence(const char* name) : OpDef(name)  
    {  
        this->Input("x")  
            .ParamType(REQUIRED)  
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})  
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})  
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});  
        this->Output("y")  
            .ParamType(REQUIRED)  
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})  
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})  
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});  

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93"); 
    }  
};  
OP_ADD(Spence);  
}  