/*!
 * \file foreach_norm.cpp
 * \brief
 */

#include <cmath>
#include <iostream>
#include "foreach_norm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "common_dtype.h"

namespace optiling {
constexpr uint32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;

constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;

class ForeachReduceTiling {
public:
    explicit ForeachReduceTiling(gert::TilingContext* context) : tilingContext(context){};
    /**
     ** function: Init
    */
    ge::graphStatus Init() {
        // Get shape, dtype information, and the total number of data.
        for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
            auto srcTensor = tilingContext->GetDynamicInputTensor(0, i);
            if (srcTensor == nullptr) {
                break;
            }
            auto srcDtype = srcTensor->GetDataType();
            // Determine whether all data types are consistent.
            if (dataType == ge::DT_UNDEFINED) {
                dataType = srcDtype;
                dataTypeSize = GetDataTypeSize(dataType);
                if (dataTypeSize == 0) {
                    dataTypeSize = BYTE_LEN_4;
                }
                elementsPerBlock = BYTE_BLOCK / dataTypeSize;
            } else if (srcDtype != dataType) {
                return ge::GRAPH_FAILED;
            }
            gert::Shape tempShape = srcTensor->GetStorageShape();
            // Make a 32-byte alignment for each Tensor
            tensorDataCountList[i] = (uint64_t)tempShape.GetShapeSize();
            if (tensorDataCountList[i] == 0) {
                isExistEmptyTensor = true;
            }
            totalBlockCount += CeilA2B(tensorDataCountList[i], elementsPerBlock);
            totalTensorCount++;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RunBigKernelTiling() {
        auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

        uint64_t ubSizePlatForm = 0;

        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

        tilingContext->SetTilingKey(GetTilingKeyVal());
        std::cout << "### tilingKey = " << GetTilingKeyVal() << std::endl;

        needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

        AssignDataToEachCore();
        DivideUbMemory(ubSizePlatForm);

        // Reduce Op Addition
        AssignTensorMiddleCountList();

        FillTilingData();

        tilingContext->SetBlockDim(needCoreNum);
        std::cout << "### BlockDim = " << tilingContext->GetBlockDim() << std::endl;

        size_t usrSize = (MAX_CORE_CONT + MAX_TENSOR_CONT) * sizeof(float);
        size_t sysWorkspaceSize = WORK_SPACE_SIZE;
        size_t *currentWorkspace = tilingContext->GetWorkspaceSizes(1);
        if (currentWorkspace==nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = usrSize + sysWorkspaceSize;

        return ge::GRAPH_SUCCESS;
    }

private:
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    uint64_t GetTilingKeyVal() {
        switch (dataType) {
            case ge::DT_FLOAT:
                return TILING_KEY_FLOAT;
            case ge::DT_FLOAT16:
                return TILING_KEY_HALF;
            case ge::DT_BF16:
                return TILING_KEY_BF16;
            default:
                return 0;
        }
    }

    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform) {
        uint32_t tempCoreNum = (uint32_t)totalBlockCount;
        if (tempCoreNum == 0) {
            tempCoreNum = 1;
        }
        if (tempCoreNum < coreNumPlatform) {
            return tempCoreNum;
        } else {
            return coreNumPlatform;
        }
    }

    void AssignDataToEachCore() {
        // Kernel the input data according to 32 byte alignment.
        // Divisible, representing the amount of data each core needs to process.
        uint64_t tempPerCoreCount = totalBlockCount / needCoreNum * elementsPerBlock;
        uint64_t remainderCount = totalBlockCount % needCoreNum;  // remainder.
        uint16_t coreIndex = 0;
        uint64_t dataCount = 0;
        uint64_t curCmpCount = 0;
        uint64_t cursorPos = 0;
        tensorStartList[coreIndex] = 0;
        tensorStartOffsetList[coreIndex] = 0;
        for (uint32_t i = 0; i < totalTensorCount; i++) {
            // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
            if (remainderCount && coreIndex < remainderCount) {
                curCmpCount = tempPerCoreCount + elementsPerBlock;
            } else {
                curCmpCount = tempPerCoreCount;
            }
            uint64_t tempRealCount = tensorDataCountList[i] - cursorPos;
            uint64_t tempCount = CeilA2B(tempRealCount, elementsPerBlock) * elementsPerBlock;
            if (dataCount + tempCount < curCmpCount) {
                dataCount += tempCount;
                cursorPos = 0;
                continue;
            }
            // dataCount >= curCmpCount, Calculate the offset
            tensorEndList[coreIndex] = i;
            cursorPos = cursorPos + curCmpCount - dataCount;
            // ReduceOp need more currect value
            tensorEndOffsetList[coreIndex] = dataCount + tempRealCount < curCmpCount ? tensorDataCountList[i] - 1 : cursorPos - 1;
            dataCount = 0;
            coreIndex++;
            if (cursorPos < tensorDataCountList[i]) {
                tensorStartList[coreIndex] = i;
                tensorStartOffsetList[coreIndex] = cursorPos;
                --i;  // The next loop continues to allocate the current tensor
            } else if (coreIndex != needCoreNum) {
                tensorStartList[coreIndex] = i + 1;
                tensorStartOffsetList[coreIndex] = 0;
                cursorPos = 0;
            }
        }
        /* The temporary count variable is not 0, which means that the last tensor is truncated,
            and you need to manually set the offset of the last core. */
        if (dataCount) {
            tensorEndList[coreIndex] = totalTensorCount - 1;
            tensorEndOffsetList[coreIndex] = tensorDataCountList[totalTensorCount - 1] - 1;
        }
    }

    void DivideUbMemory(uint64_t ubSizePlatForm) {
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 16384);
        if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 2;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ?
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }

    void AssignTensorMiddleCountList() {
        uint16_t preCoreTensorIndex = 0;
        for (uint32_t i = 1; i < needCoreNum; i++) {
            if (preCoreTensorIndex==tensorStartList[i]) {
                tensorMiddleCountList[preCoreTensorIndex]+=1;
            } else {
                if (tensorStartOffsetList[i]>0) {
                    tensorMiddleCountList[tensorStartList[i]]+=1;
                }
            }
            preCoreTensorIndex=tensorStartList[i];
        }
        uint16_t tensorMiddleStart = 0;
        for (uint32_t j = 0; j < totalTensorCount; j++) {
            tensorMiddleCountList[j]++;
            tensorMiddleStartList[j] = tensorMiddleStart;
            tensorMiddleStart += tensorMiddleCountList[j];
        }
        uint16_t coreMiddleOffset = 0;
        for (uint32_t j = 0; j<needCoreNum; j++) {
            coreMiddleOffsetList[j] = coreMiddleOffset;
            coreMiddleOffset += tensorEndList[j] - tensorStartList[j] + 1;
        }
    }

    void FillTilingData() {
        tilingData.set_inputsTensorUbSize(inputsTensorUbSize);
        tilingData.set_needCoreNum(needCoreNum);
        tilingData.set_totalTensorCount(totalTensorCount);
        tilingData.set_tensorDataCountList(tensorDataCountList);
        tilingData.set_tensorStartList(tensorStartList);
        tilingData.set_tensorEndList(tensorEndList);
        tilingData.set_tensorStartOffsetList(tensorStartOffsetList);
        tilingData.set_tensorEndOffsetList(tensorEndOffsetList);

        // Reduce Op Addition
        tilingData.set_tensorMiddleCountList(tensorMiddleCountList);
        tilingData.set_tensorMiddleStartList(tensorMiddleStartList);
        tilingData.set_coreMiddleOffsetList(coreMiddleOffsetList);

        std::cout << "### inputsTensorUbSize = " << tilingData.get_inputsTensorUbSize() << std::endl;
        std::cout << "### needCoreNum = " << tilingData.get_needCoreNum() << std::endl;
        std::cout << "### totalTensorCount = " << tilingData.get_totalTensorCount() << std::endl;
        for (uint32_t i = 0; i < tilingData.get_totalTensorCount(); ++i) {
            std::cout << "### tensorDataCountList[" << i << "] = " << tilingData.get_tensorDataCountList()[i] << std::endl;
        }
        for (uint32_t i = 0; i < tilingData.get_needCoreNum(); ++i) {
            std::cout << "### tensorStartList[" << i << "] = " << tilingData.get_tensorStartList()[i] << std::endl;
        }
        for (uint32_t i = 0; i < tilingData.get_needCoreNum(); ++i) {
            std::cout << "### tensorEndList[" << i << "] = " << tilingData.get_tensorEndList()[i] << std::endl;
        }
        for (uint32_t i = 0; i < tilingData.get_needCoreNum(); ++i) {
            std::cout << "### tensorStartOffsetList[" << i << "] = " << tilingData.get_tensorStartOffsetList()[i] << std::endl;
        }
        for (uint32_t i = 0; i < tilingData.get_needCoreNum(); ++i) {
            std::cout << "### tensorEndOffsetList[" << i << "] = " << tilingData.get_tensorEndOffsetList()[i] << std::endl;
        }
        for (uint32_t i = 0; i < tilingData.get_totalTensorCount(); ++i) {
            std::cout << "### tensorMiddleCountList[" << i << "] = " << tilingData.get_tensorMiddleCountList()[i] << std::endl;
        }
        for (uint32_t i = 0; i < tilingData.get_totalTensorCount(); ++i) {
            std::cout << "### tensorMiddleStartList[" << i << "] = " << tilingData.get_tensorMiddleStartList()[i] << std::endl;
        }
        for (uint32_t i = 0; i < tilingData.get_needCoreNum(); ++i) {
            std::cout << "### coreMiddleOffsetList[" << i << "] = " << tilingData.get_coreMiddleOffsetList()[i] << std::endl;
        }
        tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                                tilingContext->GetRawTilingData()->GetCapacity());
        tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

private:
    ForeachReduceTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;

    uint64_t inputsTensorUbSize = 0;
    uint64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorStartList[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT] = {0};
    uint64_t tensorStartOffsetList[MAX_CORE_CONT] = {0};
    uint64_t tensorEndOffsetList[MAX_CORE_CONT] = {0};
    uint64_t totalBlockCount = 0;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
    uint32_t totalTensorCount = 0;
    uint32_t needCoreNum = 0;

    bool isExistEmptyTensor = false;

    uint32_t modelCode = 0;

    uint16_t tensorMiddleCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorMiddleStartList[MAX_TENSOR_CONT] = {0};
    uint16_t coreMiddleOffsetList[MAX_CORE_CONT] = {0};
};

static ge::graphStatus Tiling4ForeachNormTiling(gert::TilingContext* context) {
    ForeachReduceTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepare4ForeachTiling(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachNorm)
.Tiling(Tiling4ForeachNormTiling)
.TilingParse<ForeachNormCompileInfo>(TilingPrepare4ForeachTiling);
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