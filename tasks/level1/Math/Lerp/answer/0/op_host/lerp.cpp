#include "lerp_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
constexpr int32_t NUM = 12;
constexpr int VAL_ZERO = 0;
constexpr int START_INDEX = 0;
constexpr int END_INDEX = 1;
constexpr int WEIGHT_INDEX = 2;

// broadcast params
uint32_t reduce1[20] = {VAL_ZERO};
uint32_t reduce2[20] = {VAL_ZERO};
uint32_t reduce3[20] = {VAL_ZERO};
uint32_t shape[20] = {VAL_ZERO};
uint32_t d = 1;
uint32_t dim = VAL_ZERO;

// tilling params
uint32_t start_length = VAL_ZERO;
uint32_t end_length = VAL_ZERO;
uint32_t weight_length = VAL_ZERO;
uint32_t total_length = VAL_ZERO;
uint32_t LERP_ALIGN_NUM = VAL_ZERO;
uint32_t tiling_size = VAL_ZERO;
uint32_t block_size = VAL_ZERO;
uint32_t core_size = VAL_ZERO;
uint32_t core_remain = VAL_ZERO;
uint32_t mode = VAL_ZERO;
// The number of vector cores, default is 1
auto aivNum = 1;

static void GetTensorShape(const gert::StorageShape* shape, std::vector<uint64_t>& inshapeVector, uint32_t outDimNum) {
    int n = outDimNum;
    for (int j = shape->GetStorageShape().GetDimNum() - 1; j >= 0; --j) {
        inshapeVector[--n] = shape->GetStorageShape().GetDim(j);
    }
}

static void GetBroadCastParams(std::vector<uint64_t> inshapeVector1, std::vector<uint64_t> inshapeVector2,
                                std::vector<uint64_t> inshapeVector3, gert::Shape outshape){
    // Broadcast based on the GreaterEqual operator from 0x8C
    // Caculate broadcast params
    for(int i=0;i<outshape.GetDimNum();i++){
        shape[i] = outshape.GetDim(i);
        d *= outshape.GetDim(i);
        if(inshapeVector1[i] != outshape.GetDim(i)) reduce1[i] = 1;
        if(inshapeVector2[i] != outshape.GetDim(i)) reduce2[i] = 1;
        if(inshapeVector3[i] != outshape.GetDim(i)) reduce3[i] = 1;
    }

    dim = outshape.GetDimNum();
    for(int i=dim-1;i>=1;i--){
        if(!reduce1[i - 1] && !reduce2[i - 1] && !reduce3[i - 1] && !reduce1[i] && !reduce2[i] && !reduce3[i]){
            dim--;
            shape[i - 1] *= shape[i];
        }else{
            break;
        }
    }
    if(reduce1[dim - 1] || reduce2[dim - 1] || reduce3[dim - 1]){
        shape[dim] = 1;
        dim++;
    }
}

static void GetTillingParams(gert::TilingContext* context) {
    // caculate tilling params
    uint32_t sizeofdatatype;
    uint64_t ub_size;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    aivNum = ascendcPlatform.GetCoreNum();

    const int INPUT_TENSOR_COUNT = 3;
    total_length = 0;
    for (int i = 0; i < INPUT_TENSOR_COUNT; ++i) {
        total_length = std::max<uint32_t>(total_length, context->GetInputTensor(i)->GetShapeSize());
    }

    start_length = context->GetInputShape(START_INDEX)->GetStorageShape().GetShapeSize();
    end_length = context->GetInputShape(END_INDEX)->GetStorageShape().GetShapeSize();
    weight_length = context->GetInputShape(WEIGHT_INDEX)->GetStorageShape().GetShapeSize();
    if(weight_length != 1) {
        mode = 1;
    }

    auto dt = context->GetInputDesc(0)->GetDataType();
    if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        // Data type is 16-bit (2 bytes)
        sizeofdatatype = 2;
    }else{
        // Data type is 32-bit (4 bytes)
        sizeofdatatype = 4;
    }
    LERP_ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    const int UB_DIV_COEF = 2;
    tiling_size = ((ub_size) / BLOCK_SIZE / UB_DIV_COEF) / NUM;
    // 8 blocks for 256-byte alignment
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
    block_size = tiling_size * LERP_ALIGN_NUM;
    aivNum = (aivNum < total_length / block_size) ? aivNum : (total_length / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;
    if(aivNum == 0){
        aivNum = 1;
    }    
    core_size = (total_length / aivNum) / (LERP_ALIGN_NUM * 8) * (LERP_ALIGN_NUM * 8); // 8 blocks for 256-byte alignment
    core_remain = total_length - aivNum * core_size;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LerpTilingData tiling;

    // Get input tensors' shape
    auto outshape = context->GetOutputShape(0)->GetOriginShape();
    uint32_t outDimNum = outshape.GetDimNum();
    const gert::StorageShape* shape1 = context->GetInputShape(0);
    const gert::StorageShape* shape2 = context->GetInputShape(1);
    const gert::StorageShape* shape3 = context->GetInputShape(2);
    std::vector<uint64_t> inshapeVector1(outDimNum, 1);
    std::vector<uint64_t> inshapeVector2(outDimNum, 1);
    std::vector<uint64_t> inshapeVector3(outDimNum, 1);

    GetTensorShape(shape1, inshapeVector1, outDimNum);
    GetTensorShape(shape2, inshapeVector2, outDimNum);
    GetTensorShape(shape3, inshapeVector3, outDimNum);

    // Check if broadcasting is needed
    bool flag = false;
    for(int i=0;i<outshape.GetDimNum();i++){
        if(shape1->GetStorageShape().GetDim(i) != shape2->GetStorageShape().GetDim(i) ||shape1->GetStorageShape().GetDim(i) != shape3->GetStorageShape().GetDim(i)) flag = true;
    }
    if(flag){
        GetBroadCastParams(inshapeVector1, inshapeVector2, inshapeVector3, outshape);
        tiling.set_shape(shape);
        tiling.set_reduce1(reduce1);
        tiling.set_reduce2(reduce2);
        tiling.set_reduce3(reduce3);
        tiling.set_dim(dim);
    }

    GetTillingParams(context);
    tiling.set_start_length(start_length);
    tiling.set_end_length(end_length);
    tiling.set_weight_length(weight_length);
    tiling.set_total_length(total_length);
    tiling.set_ALIGN_NUM(LERP_ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);
    tiling.set_mode(mode);

    context->SetBlockDim(aivNum);
    if(start_length==total_length && end_length==total_length && (weight_length==total_length||mode==1)){
        context->SetTilingKey(0);
    }else{
        context->SetTilingKey(1);
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    std::cout << "*******************START*******************" << std::endl;
    std::cout << "coreNum = " << aivNum << std::endl;
    std::cout << "total_length = " << tiling.get_total_length() << std::endl;
    std::cout << "start_length = " << tiling.get_start_length() << std::endl;
    std::cout << "end_length = " << tiling.get_end_length() << std::endl;
    std::cout << "weight_length = " << tiling.get_weight_length() << std::endl;
    std::cout << "ALIGN_NUM = " << tiling.get_ALIGN_NUM() << std::endl;
    std::cout << "tiling_size = " << tiling.get_tiling_size() << std::endl;
    std::cout << "block_size = " << tiling.get_block_size() << std::endl;
    std::cout << "core_size = " << tiling.get_core_size() << std::endl;
    std::cout << "core_remain = " << tiling.get_core_remain() << std::endl;
    std::cout << "mode = " << tiling.get_mode() << std::endl;
    std::cout << "dim = " << tiling.get_dim() << std::endl;
    std::cout << "*******************END*******************" << std::endl;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Lerp : public OpDef {
public:
    explicit Lerp(const char* name) : OpDef(name)
    {
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910_93")
                          .AddConfig("ascend910b");
    }
};

OP_ADD(Lerp);
}