#include <algorithm>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"

namespace ops {
class ForeachAddScalar: public OpDef {
public:
explicit ForeachAddScalar(const char* name) : OpDef(name) {
std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16};
std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
std::vector<ge::DataType> scalar_tensor_dtype_list;
std::for_each(tensor_dtype_list.cbegin(), tensor_dtype_list.cend(), [&scalar_tensor_dtype_list](ge::DataType dtype){scalar_tensor_dtype_list.push_back(DtypeScalarToTensor2(dtype));});
this->Input("x")
.ParamType(DYNAMIC)
.DataType(tensor_dtype_list)
.Format(format_list)
.UnknownShapeFormat(format_list)
.AutoContiguous();
this->Input("scalar")
.ParamType(REQUIRED)
.DataType(scalar_tensor_dtype_list)
.Format(format_list)
.UnknownShapeFormat(format_list);
this->Output("y")
.ParamType(DYNAMIC)
.DataType(tensor_dtype_list)
.Format(format_list)
.UnknownShapeFormat(format_list)
.AutoContiguous();
this->AICore().AddConfig("ascend910b");
this->AICore().AddConfig("ascend910_93");
}
};
OP_ADD(ForeachAddScalar);
}  // namespace ops