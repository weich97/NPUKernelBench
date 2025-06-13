#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "foreach_mul_list_tiling.h"
#include "foreach_proto_utils.h"
#include "common_dtype.h"

namespace optiling {
    constexpr uint64_t TILING_HALF_N_SCALAR = 14;
    constexpr uint64_t TILING_FLOAT_N_SCALAR = 4;
    constexpr uint64_t TILING_INT_N_SCALAR = 4;
    constexpr uint64_t TILING_BF16_N_SCALAR = 14;
    constexpr uint32_t TILING_FLOAT_ERF = 5;
    constexpr uint32_t TILING_HALF_ERF = 12;
    constexpr uint8_t BYTE_PER_BLOCK = 32;
    constexpr uint64_t WORK_SPACE_SIZE = 32;// foreach(vector) not need workspace
    constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;
    constexpr uint32_t BINARY_LIST_UB_DIVIDER = 6;

class ForeachCommonTiling {
public:
    explicit ForeachCommonTiling(gert::TilingContext* context) : tilingContext(context){};
    /**
     ** function: Init
    */
    ge::graphStatus Init() {
        int dynamicIdx = 0;
        for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
            auto srcTensor = tilingContext->GetDynamicInputTensor(dynamicIdx, i);
            if (srcTensor == nullptr) {
                break;
            }

            auto temp = tilingContext->GetInputDesc(0);
            if (temp == nullptr) {
                return ge::GRAPH_FAILED;
            }

            auto srcDtype = temp->GetDataType();

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
            tensorDataCountList[i] = tempShape.GetShapeSize();
            totalDataCount += tensorDataCountList[i];
            totalTensorCount++;
        }
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: RunBigKernelTiling
    */
    ge::graphStatus RunBigKernelTiling() {
        auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

        uint64_t ubSizePlatForm = 0;
        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

        tilingContext->SetTilingKey(GetTilingKeyByDtypeOnly(dataType));

        uint32_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

        AssignDataToEachCore(needCoreNum);
        DivideUbMemory(ubSizePlatForm);
        FillTilingData();
        tilingContext->SetBlockDim(needCoreNum);
        size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
        if (workspaces == nullptr) {
            return ge::GRAPH_FAILED;
        }
        workspaces[0] = WORK_SPACE_SIZE;

        return ge::GRAPH_SUCCESS;
    }

private:
    /**
     ** function: CeilA2B
    */
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    /**
     ** function: GetTilingN
    */
    uint64_t GetTilingN() {
        switch (dataType) {
            case ge::DT_FLOAT:
                return TILING_FLOAT_N_SCALAR;
            case ge::DT_FLOAT16:
                return TILING_HALF_N_SCALAR;
            case ge::DT_INT32:
                return TILING_INT_N_SCALAR;
            case ge::DT_BF16:
                return TILING_BF16_N_SCALAR;
            default:
                return TILING_HALF_N_SCALAR;
        }
    }

    /**
     ** function: GetNeedCoreNum
    */
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform) {
        uint32_t tempCoreNum = (uint32_t)CeilA2B(totalDataCount, elementsPerBlock);
        if (tempCoreNum == 0) {
            tempCoreNum = 1;
        }
        if (tempCoreNum < coreNumPlatform) {
            return tempCoreNum;
        } else {
            return coreNumPlatform;
        }
    }

    /**
     ** function: AssignDataToEachCore
    */
    void AssignDataToEachCore(int64_t needCoreNum) {
        // Kernel the input data according to 32 byte alignment.
        int64_t blockCount = CeilA2B(totalDataCount, elementsPerBlock);
        // Divisible, representing the amount of data each core needs to process.
        if (needCoreNum == 0) {
            needCoreNum = 1;
        }
        int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock;
        int64_t remainderCount = blockCount % needCoreNum;  // remainder.
        uint16_t coreIndex = 0;
        int64_t dataCount = 0;
        int64_t curCmpCount = 0;
        int64_t cursorPosition = 0;
        tensorStartList[coreIndex] = 0;
        tensorStartOffsetList[coreIndex] = 0;
        for (uint16_t i = 0; i < totalTensorCount; i++) {
            // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
            if (remainderCount && coreIndex < remainderCount) {
                curCmpCount = tempPerCoreCount + elementsPerBlock;
            } else {
                curCmpCount = tempPerCoreCount;
            }
            int64_t tempCount = tensorDataCountList[i] - cursorPosition;

            if (dataCount + tempCount < curCmpCount) {
                dataCount += tempCount;
                cursorPosition = 0;
                continue;
            }
            // dataCount >= curCmpCount, Calculate the offset
            tensorEndList[coreIndex] = i;
            cursorPosition = cursorPosition + curCmpCount - dataCount;
            tensorEndOffsetList[coreIndex] = cursorPosition - 1;
            dataCount = 0;
            coreIndex++;
            if (cursorPosition < tensorDataCountList[i]) {
                tensorStartList[coreIndex] = i;
                tensorStartOffsetList[coreIndex] = cursorPosition;
                --i;  // The next loop continues to allocate the current tensor
            } else if (coreIndex != needCoreNum) {
                tensorStartList[coreIndex] = i + 1;
                tensorStartOffsetList[coreIndex] = 0;
                cursorPosition = 0;
            }
        }
        /* The temporary count variable is not 0, which means that the last tensor is truncated,
            and you need to manually set the offset of the last core. */
        if (dataCount) {
            tensorEndList[coreIndex] = totalTensorCount - 1;
            tensorEndOffsetList[coreIndex] = tensorDataCountList[totalTensorCount - 1] - 1;
        }
    }

    /**
     ** funtion: DivideUbMemory
    */
    void DivideUbMemory(uint64_t ubSizePlatForm) {
        // The remaining UB size is split in six, double buffer enabled, and rounded down 32 bytes.
        // foreach_div_list/minimum_list/mul_list/sub_list
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
        if (dataType == ge::DT_BF16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / BINARY_LIST_UB_DIVIDER;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ?
            canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
            canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }

    /**
     ** function: FillTilingData
    */
    void FillTilingData() {
        tilingData.set_inputsTensorUbSize(inputsTensorUbSize);
        tilingData.set_tensorDataCountList(tensorDataCountList);
        tilingData.set_tensorStartList(tensorStartList);
        tilingData.set_tensorEndList(tensorEndList);
        tilingData.set_tensorStartOffsetList(tensorStartOffsetList);
        tilingData.set_tensorEndOffsetList(tensorEndOffsetList);

        tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                                tilingContext->GetRawTilingData()->GetCapacity());
        tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

private:
    ForeachCommonTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;

    uint64_t inputsTensorUbSize = 0;
    int64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorStartList[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT] = {0};
    int64_t tensorStartOffsetList[MAX_CORE_CONT] = {0};
    int64_t tensorEndOffsetList[MAX_CORE_CONT] = {0};
    int64_t totalDataCount = 0;
    uint8_t dataTypeSize = 4;
    uint8_t elementsPerBlock = 0;
    uint16_t totalTensorCount = 0;
};

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

}

namespace ops {
// 模型的参数使用如下宏进行定义
FOREACH_OPDEF_WITH_TILING(HOST_CONFIG, BINARY_LIST, MulList, optiling::TilingFunc, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16)
}  // namespace ops

