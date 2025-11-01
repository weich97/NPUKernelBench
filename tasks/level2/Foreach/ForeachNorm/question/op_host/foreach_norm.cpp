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
 class ForeachNorm : public OpDef {
 public:
     explicit ForeachNorm(const char* name) : OpDef(name)
     {
         // The following sections are the result of macro expansion.
         // Specifically, this structure aligns with FOREACH_BINARY_SCALAR_PARAM,
         // which involves FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE and FOREACH_SCALAR_DTYPE_PREPARE.
         //
         // For example, if we consider a hypothetical macro call:
         // FOREACH_OPDEF(HOST_CONFIG, BINARY_SCALAR, Norm, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16)
         //
         // Step 1: FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE:
         // std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16};
         // std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
         // (This results in the DataType and Format for "x" and "y" below)
         this->Input("x")
             .ParamType(DYNAMIC)
             .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .AutoContiguous();

         // Step 2: FOREACH_SCALAR_DTYPE_PREPARE:
         // std::vector<ge::DataType> scalar_dtype_list;
         // std::for_each(tensor_dtype_list.cbegin(), tensor_dtype_list.cend(), [&scalar_dtype_list](ge::DataType dtype){scalar_dtype_list.push_back(DtypeTensor2Scalar(dtype));});
         //
         // Note: Based on the DtypeTensor2Scalar function provided, if tensor_dtype_list was
         // {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16},
         // then scalar_dtype_list would be {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}.
         // However, the provided code's DataType for "scalar" is {ge::DT_FLOAT, ge::DT_INT64, ...}.
         // This implies the original tensor_dtype_list used for scalar generation would need to be
         // something like {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16, ge::DT_INT32}
         // to produce the alternating DT_FLOAT/DT_INT64 types.
         // This expanded code preserves the original content as requested, representing the final state after expansion.
         this->Input("scalar")
             .Scalar()
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
         this->Output("y")
             .ParamType(DYNAMIC)
             .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .AutoContiguous();
 
         // Expansion of FOREACH_OPDEF_END_HOST_CONFIG(Norm)
         this->AICore().AddConfig("ascend910b");
         this->AICore().AddConfig("ascend910_93");
     }
 };
 
 // Expansion of OP_ADD(ForeachNorm);
 OP_ADD(ForeachNorm);
 }