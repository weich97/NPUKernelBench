/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rms_norm.cpp
 * \brief
 */
#include <iostream>
#include "register/op_def_registry.h"
#include "rms_norm_tiling.h"
#include "tiling/tiling_api.h"

using namespace ge;

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(op_name, ...)   std::printf(op_name, ##__VA_ARGS__)
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)        \
  do {                                                               \
    std::printf("op[%s], %s", op_name, err_msg);                     \
  } while (0)
}  // namespace ops
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

namespace optiling {
namespace {
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;
constexpr uint32_t UB_FACTOR_B16 = 12288;
constexpr uint32_t GEMMA_UB_FACTOR_B16 = 10240;
constexpr uint32_t GEMMA_UB_FACTOR_B32 = 8192;
constexpr uint32_t UB_FACTOR_B32 = 10240;
constexpr uint32_t BLOCK_ALIGN_NUM = 16;
constexpr uint32_t FLOAT_BLOCK_ALIGN_NUM = 8;
constexpr uint32_t FLOAT_PER_REAPEAT = 64;
constexpr uint32_t BYTE_SIZE_2_BLOCK_ALIGN_NUM = 16;
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t GAMMA_INDEX = 1;
constexpr uint32_t FLOAT_BYTE_SIZE = 4;
constexpr uint32_t B16_BYTE_SIZE = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t UB_USED = 1024;
constexpr uint32_t UB_COUNTS = 3;
constexpr uint32_t FLOAT_UB_COUNTS = 2;
constexpr uint32_t UB_COUNTS_X_SQX = 2;
constexpr uint32_t UB_COUNTS_GAMMA = 1;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t MASK_64 = 64;
constexpr uint32_t SOC_WEIGHT = 1000;
constexpr uint32_t ALIGN_WEIGHT = 100;
constexpr uint32_t DTYPE_WEIGHT = 10;
constexpr uint32_t DIV_TO_HALF = 2;
constexpr uint32_t SMALL_REDUCE_NUM = 2000;
constexpr uint32_t MODE_NORMAL = 0;
constexpr uint32_t MODE_SPLIT_D = 1;
constexpr uint32_t MODE_MERGE_N = 2;
constexpr uint32_t MODE_SINGLE_ROW = 3;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_X = 1;
constexpr size_t MIN_DIM_GAMMA = 1;
constexpr size_t SYS_WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr uint64_t NUM_260 = 260;

RMSNormTilingData tilingData;

template <typename T>
static T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

void SetByDtype(ge::DataType dataType, uint32_t &dtypeKey, uint32_t &dataPerBlock)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dtypeKey = DTYPE_KEY_FP16;
            dataPerBlock = BYTE_SIZE_2_BLOCK_ALIGN_NUM;
            break;
        case ge::DT_BF16:
            dtypeKey = DTYPE_KEY_BF16;
            dataPerBlock = BYTE_SIZE_2_BLOCK_ALIGN_NUM;
            break;
        default:
            dtypeKey = DTYPE_KEY_FP32;
            dataPerBlock = FLOAT_BLOCK_ALIGN_NUM;
            break;
    }
}

int32_t FindPowerTwo(int32_t n)
{
    // Set all the bits after the first 1 in the binary of n to 1,
    // then add 1 and shift one bit to the right to find max power of 2 no more than n (32 bit)
    n |= n >> 1;  // Set the first digit of n's binary to 1
    n |= n >> 2;  // Set the first two bits of n's binary to 1
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;  // Set the first 32 bits of n's binary to 1
    return (n + 1) >> 1;
}

void OnceReduceMaxColsAlign64(uint64_t &onceReduceMaxCols, uint32_t &mask, uint32_t &leftNum)
{
    uint32_t nowCols = FindPowerTwo(onceReduceMaxCols);
    leftNum = static_cast<uint32_t>(onceReduceMaxCols) - nowCols;
    while (nowCols > FLOAT_BLOCK_ALIGN_NUM) {
        nowCols = nowCols / DIV_TO_HALF;
    }
    mask = nowCols;
}
}  // namespace

static bool CheckInputShape4RmsNorm(const gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::StorageShape *gammaShape = context->GetInputShape(1);
    const gert::StorageShape *yShape = context->GetOutputShape(0);
    const gert::StorageShape *rstdShape = context->GetOutputShape(1);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();
    size_t rstdDimNum = rstdShape->GetStorageShape().GetDimNum();

    return true;
}

bool CalMixDtypeTiling(uint32_t &modeKey, uint64_t &rowFactor, uint64_t &ubFactor, const RMSNormTilingInfo &rmsTilInfo)
{
    uint64_t ubSize = rmsTilInfo.ubSize;
    uint64_t numColAlign =
        CeilDiv(rmsTilInfo.numCol, static_cast<uint64_t>(BYTE_SIZE_2_BLOCK_ALIGN_NUM)) * BYTE_SIZE_2_BLOCK_ALIGN_NUM;

    // Cal buffer size
    uint64_t oneRowXYGammaBufSize = B16_BYTE_SIZE * 2 * numColAlign            // x&y
                                    + FLOAT_BYTE_SIZE * 1 * numColAlign;       // gamma, used numColAlign
    uint64_t oneRowRstdBufSize = FLOAT_BYTE_SIZE * 1 * FLOAT_BLOCK_ALIGN_NUM;  // rstd
    uint64_t oneRowtmpBufSize = FLOAT_BYTE_SIZE * 2 * numColAlign;             // xFp32&sqx, used numColAlign
    uint64_t oneRowReduceBufSize = FLOAT_BYTE_SIZE * 1 * FLOAT_PER_REAPEAT;    // buffer for reduce

    uint64_t mutiRowRstdBufSize =
        FLOAT_BYTE_SIZE * 1 * FLOAT_PER_REAPEAT;  // rowFactor = FLOAT_PER_REAPEAT, bigger than oneRowRstdBufSize.

    // 1. Mode MergeN
    uint64_t oneRowBufSize = oneRowXYGammaBufSize + oneRowtmpBufSize + oneRowRstdBufSize + oneRowReduceBufSize;
    if (numColAlign <= SMALL_REDUCE_NUM && rmsTilInfo.isSoc910B) {
        modeKey = MODE_MERGE_N;
        rowFactor = ubSize / oneRowBufSize;  // oneRowBufSize not be zero, div without check.
        ubFactor = rowFactor * numColAlign;
        return true;
    }

    // 2. Mode Normal
    uint64_t bufSizeNew = oneRowXYGammaBufSize + oneRowtmpBufSize + mutiRowRstdBufSize + oneRowReduceBufSize;
    if (bufSizeNew < ubSize) {
        modeKey = MODE_NORMAL;
        rowFactor = FLOAT_PER_REAPEAT;
        ubFactor = numColAlign;
        return true;
    }

    // 3. Mode SingleRow
    uint64_t oneRowXYGammaBufSizeNew = B16_BYTE_SIZE * 1 * numColAlign       // x
                                       + FLOAT_BYTE_SIZE * 1 * numColAlign;  // gamma
    uint64_t oneRowBufSizeNew = oneRowXYGammaBufSizeNew + oneRowtmpBufSize + mutiRowRstdBufSize + oneRowReduceBufSize;
    if (oneRowBufSizeNew < ubSize) {
        modeKey = MODE_SINGLE_ROW;
        rowFactor = FLOAT_PER_REAPEAT;
        ubFactor = numColAlign;
        return true;
    }

    // 4. Mode SplitD
    modeKey = MODE_SPLIT_D;
    rowFactor = FLOAT_PER_REAPEAT;
    uint64_t oneColSize = B16_BYTE_SIZE * 2       // x&y
                          + FLOAT_BYTE_SIZE * 3;  // gamma&xFp32&sqx
    uint64_t mutiRowSumBufSize = FLOAT_BYTE_SIZE * FLOAT_BLOCK_ALIGN_NUM * rowFactor;
    uint64_t notColBufSizeSum = mutiRowRstdBufSize + oneRowReduceBufSize + mutiRowSumBufSize;
    uint64_t tmpCol;
    if (ubSize > notColBufSizeSum) {
        tmpCol = (ubSize - notColBufSizeSum) / oneColSize;  // oneColSize not be zero, div without check.
    } else {
        return false;
    }
    tmpCol = (tmpCol / BLOCK_ALIGN_NUM) * BLOCK_ALIGN_NUM;
    ubFactor = tmpCol;
    return true;
}

static ge::graphStatus Tiling4RmsNorm(gert::TilingContext *context)
{
    RMSNormTilingData tiling;
    auto ptrCompileInfo = reinterpret_cast<const Tiling4RmsNormCompileInfo *>(context->GetCompileInfo());
    uint32_t numCore;
    uint64_t ubSize;
    platform_ascendc::SocVersion curSocVersion;
    if (nullptr == ptrCompileInfo) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        curSocVersion = ascendc_platform.GetSocVersion();
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
        curSocVersion = ptrCompileInfo->curSocVersion;
    }
    bool isSoc910B = (curSocVersion == platform_ascendc::SocVersion::ASCEND910B);
    const gert::Shape x_shape = context->GetInputShape(X_INDEX)->GetStorageShape();

    const gert::Shape gamma_shape = context->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    std::string opType(context->GetNodeType());
    auto attrs = context->GetAttrs();
    const float *epsilon = attrs->GetFloat(0);
    uint64_t numCol = gamma_shape.GetShapeSize();
    float avgFactor = (numCol == 0) ? 0 : 1.0 / numCol;
    size_t xDimNum = x_shape.GetDimNum();
    size_t gammaDimNum = gamma_shape.GetDimNum();
    uint64_t numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= x_shape.GetDim(i);
    }

    bool isMixDtype = false;
    auto xDesc = context->GetInputDesc(0);
    auto gammaDesc = context->GetInputDesc(1);
    auto xDataType = xDesc->GetDataType();
    auto gammaDataType = gammaDesc->GetDataType();
    uint32_t xDtypeKey = DTYPE_KEY_FP16;

    size_t usrSize = 256;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    uint32_t SocVersion = 0;
    uint64_t numColAlign = 0;
    uint64_t onceReduceMaxCols;
    uint32_t mask{0};
    uint32_t lastMask{0};
    uint32_t leftNum{0};
    uint32_t lastLeftNum{0};
    uint64_t colAlign = 0;
    uint32_t rstdSize{0};
    uint64_t blockFactor;
    uint64_t ubFactor;
    uint64_t rowFactor;
    uint32_t modeKey = MODE_NORMAL;

    ubSize = ubSize - UB_USED;

    uint32_t dataPerBlock;
    SetByDtype(xDataType, xDtypeKey, dataPerBlock);
    if (xDataType != gammaDataType && gammaDataType == ge::DT_FLOAT) {
        // only support x fp16/bf16 gamma fp32
        isMixDtype = true;
    }

    numColAlign = CeilDiv(numCol, static_cast<uint64_t>(dataPerBlock)) * dataPerBlock;

    if (curSocVersion == platform_ascendc::SocVersion::ASCEND910) {
        SocVersion = 1;
        colAlign = numColAlign == numCol ? 1 : 0;
        int64_t ubDataNum = ubSize / FLOAT_BYTE_SIZE;
        if (static_cast<uint64_t>(ubDataNum) > numColAlign) {
            rowFactor = (ubDataNum - numColAlign) /
                         (BUFFER_NUM * FLOAT_PER_REAPEAT + numColAlign * UB_COUNTS_X_SQX * BUFFER_NUM);
        } else {
            rowFactor = 0;
        }
        blockFactor = CeilDiv(numRow, static_cast<uint64_t>(numCore));
        blockFactor = CeilDiv(blockFactor, static_cast<uint64_t>(FLOAT_BLOCK_ALIGN_NUM)) * FLOAT_BLOCK_ALIGN_NUM;
        blockFactor = std::max(static_cast<uint64_t>(dataPerBlock), blockFactor);
        blockFactor = std::min(blockFactor, numRow);

        if (rowFactor > BYTE_SIZE_2_BLOCK_ALIGN_NUM) {
            rowFactor = rowFactor / BYTE_SIZE_2_BLOCK_ALIGN_NUM * BYTE_SIZE_2_BLOCK_ALIGN_NUM;
        }

        uint32_t useCoreNum = CeilDiv(numRow, blockFactor);
        context->SetBlockDim(useCoreNum);
        ubFactor = 1;
        modeKey = 0;
        if (rowFactor == 0) {
            modeKey = 1;
            uint32_t bufferFactor = (dataPerBlock / FLOAT_BLOCK_ALIGN_NUM);

            onceReduceMaxCols = bufferFactor * (ubDataNum - MASK_64 * FLOAT_BLOCK_ALIGN_NUM - MASK_64) /
                                   (UB_COUNTS + bufferFactor * FLOAT_UB_COUNTS) / dataPerBlock * dataPerBlock;
            onceReduceMaxCols = onceReduceMaxCols / dataPerBlock * dataPerBlock;
            ubFactor = CeilDiv(numColAlign, static_cast<uint64_t>(onceReduceMaxCols));
            ubFactor = CeilDiv(numColAlign, static_cast<uint64_t>(ubFactor));
            ubFactor = (ubFactor + dataPerBlock - 1) / dataPerBlock * dataPerBlock;
            OnceReduceMaxColsAlign64(ubFactor, mask, leftNum);
            uint32_t repeatTime = CeilDiv(numCol, static_cast<uint64_t>(ubFactor));
            uint64_t colTail = numCol - (repeatTime - 1) * ubFactor;
            OnceReduceMaxColsAlign64(colTail, lastMask, lastLeftNum);
        } else {
            onceReduceMaxCols = numCol;
            OnceReduceMaxColsAlign64(onceReduceMaxCols, mask, leftNum);
        }
        rowFactor = std::min(rowFactor, blockFactor);
        rstdSize = rowFactor * FLOAT_PER_REAPEAT * FLOAT_BYTE_SIZE;
    } else if (isMixDtype) {
        blockFactor = CeilDiv(numRow, static_cast<uint64_t>(numCore));
        uint32_t useCoreNum = CeilDiv(numRow, blockFactor);
        context->SetBlockDim(useCoreNum);

        RMSNormTilingInfo rmsNormTilingInfo;
        rmsNormTilingInfo.ubSize = ubSize;
        rmsNormTilingInfo.numCol = numCol;
        rmsNormTilingInfo.numRow = numRow;
        rmsNormTilingInfo.isSoc910B = isSoc910B;
        bool res = CalMixDtypeTiling(modeKey, rowFactor, ubFactor, rmsNormTilingInfo);
    } else {
        ubFactor = (xDtypeKey == DTYPE_KEY_FP32) ? UB_FACTOR_B32 : UB_FACTOR_B16;
        blockFactor = 1;
        uint64_t tileNum = CeilDiv(numRow, numCore * blockFactor);
        blockFactor *= tileNum;
        uint32_t useCoreNum = CeilDiv(numRow, blockFactor);

        context->SetBlockDim(useCoreNum);

        rowFactor = FLOAT_PER_REAPEAT;

        if (numColAlign > ubFactor) {
            modeKey = MODE_SPLIT_D;
        }

        if (numColAlign <= SMALL_REDUCE_NUM && curSocVersion == platform_ascendc::SocVersion::ASCEND910B) {
            modeKey = MODE_MERGE_N;
        }

        if (modeKey == MODE_SPLIT_D) {
            uint64_t colTileNum = CeilDiv(numCol, ubFactor);
            ubFactor = CeilDiv(numCol, colTileNum * BLOCK_ALIGN_NUM) * BLOCK_ALIGN_NUM;
            while (numCol % ubFactor != 0 && numCol % ubFactor < BLOCK_ALIGN_NUM) {
                ubFactor -= BLOCK_ALIGN_NUM;
                if (ubFactor == 0) {
                    std::printf("Tiling split last dim failed, please check.");
                    return ge::GRAPH_FAILED;
                }
            }
        }
        if (modeKey == MODE_MERGE_N) {
            uint64_t numColAlignWeight = (xDtypeKey == DTYPE_KEY_FP32) ? 16 : 14;
            rowFactor = ubSize / (numColAlign * numColAlignWeight + NUM_260);
            if (curSocVersion == platform_ascendc::SocVersion::ASCEND310P) {
                rowFactor = rowFactor / FLOAT_BLOCK_ALIGN_NUM * FLOAT_BLOCK_ALIGN_NUM;  // BroadCast need 32B Align
            }
            ubFactor = rowFactor * numColAlign;

            if (ubFactor == 0) {
                return ge::GRAPH_FAILED;
            }
        }
    }

    uint32_t tilingKey = (modeKey == 0) ? colAlign * ALIGN_WEIGHT + SocVersion * SOC_WEIGHT + modeKey : modeKey;
    if ((curSocVersion == platform_ascendc::SocVersion::ASCEND910) && (modeKey == 0)) {
        tilingKey = tilingKey + xDtypeKey * DTYPE_WEIGHT;
    }
    context->SetTilingKey(tilingKey);

    tiling.set_num_row(numRow);
    tiling.set_num_col(numCol);
    tiling.set_num_col_align(numColAlign);

    tiling.set_reduce_mask(mask);
    tiling.set_last_reduce_mask(lastMask);

    tiling.set_block_factor(blockFactor);
    tiling.set_row_factor(rowFactor);
    tiling.set_ub_factor(ubFactor);
    tiling.set_left_num(leftNum);
    tiling.set_last_left_num(lastLeftNum);

    tiling.set_epsilon(*epsilon);
    tiling.set_avg_factor(avgFactor);
    tiling.set_rstd_size(rstdSize);
    tiling.set_is_gemma(0);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    std::cout << "*******************START*******************" << std::endl;
    std::cout << "coreNum = " << context->GetBlockDim() << std::endl;
    std::cout << "num_row = " << tiling.get_num_row() << std::endl;
    std::cout << "num_col = " << tiling.get_num_col() << std::endl;
    std::cout << "num_col_align = " << tiling.get_num_col_align() << std::endl;
    std::cout << "block_factor = " << tiling.get_block_factor() << std::endl;
    std::cout << "row_factor = " << tiling.get_row_factor() << std::endl;
    std::cout << "ub_factor = " << tiling.get_ub_factor() << std::endl;
    std::cout << "reduce_mask = " << tiling.get_reduce_mask() << std::endl;
    std::cout << "left_num = " << tiling.get_left_num() << std::endl;
    std::cout << "last_reduce_mask = " << tiling.get_last_reduce_mask() << std::endl;
    std::cout << "last_left_num = " << tiling.get_last_left_num() << std::endl;
    std::cout << "rstd_size = " << tiling.get_rstd_size() << std::endl;
    std::cout << "epsilon = " << tiling.get_epsilon() << std::endl;
    std::cout << "avg_factor = " << tiling.get_avg_factor() << std::endl;
    std::cout << "is_gemma = " << tiling.get_is_gemma() << std::endl;
    std::cout << "*******************END*******************" << std::endl;
    return ge::GRAPH_SUCCESS;
}

void setPlateCoreInfo(gert::TilingContext *context, uint32_t &numCore, uint64_t &ubSize)
{
    auto ptrCompileInfo = reinterpret_cast<const Tiling4RmsNormCompileInfo *>(context->GetCompileInfo());
    if (nullptr == ptrCompileInfo) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
    }
}

static bool setEpsTiling(const gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    const float *epsilon = attrs->GetFloat(0);
    tilingData.set_epsilon(*epsilon);

    return true;
}

static bool getUbFactor(
    const uint32_t dataPerBlock, const uint32_t xDtypeKey, const uint32_t numCore, gert::TilingContext *context)
{
    const gert::Shape x_shape = context->GetInputShape(X_INDEX)->GetStorageShape();
    const gert::Shape gamma_shape = context->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    std::string opType(context->GetNodeType());

    uint64_t numCol = gamma_shape.GetShapeSize();
    float avgFactor = (static_cast<float>(numCol) == 0.0f) ? 0.0f : (1.0f / static_cast<float>(numCol));
    size_t xDimNum = x_shape.GetDimNum();
    size_t gammaDimNum = gamma_shape.GetDimNum();
    uint64_t numRow = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= x_shape.GetDim(i);
    }

    uint64_t numColAlign =
        CeilDiv(numCol, static_cast<uint64_t>(dataPerBlock)) * static_cast<uint64_t>(dataPerBlock);
    uint64_t blockFactor = 1;
    uint64_t ubFactor = (xDtypeKey == DTYPE_KEY_FP32) ? GEMMA_UB_FACTOR_B32 : GEMMA_UB_FACTOR_B16;
    uint32_t modeKey = MODE_NORMAL;

    uint64_t tileNum = CeilDiv(numRow, numCore * blockFactor);
    blockFactor *= tileNum;
    uint32_t useCoreNum = CeilDiv(numRow, blockFactor);

    if (numColAlign > ubFactor) {
        modeKey = MODE_SPLIT_D;
        uint64_t colTileNum = CeilDiv(numCol, ubFactor);
        ubFactor = CeilDiv(numCol, colTileNum * BLOCK_ALIGN_NUM) * BLOCK_ALIGN_NUM;
        while ((numCol % ubFactor) != (static_cast<uint64_t>(0)) &&
               (numCol % ubFactor) < (static_cast<uint64_t>(BLOCK_ALIGN_NUM))) {
            ubFactor -= (static_cast<uint64_t>(BLOCK_ALIGN_NUM));
            if (ubFactor == static_cast<uint64_t>(0)) {
                return false;
            }
        }
    }

    context->SetBlockDim(useCoreNum);
    context->SetTilingKey(modeKey);

    tilingData.set_num_col_align(numColAlign);
    tilingData.set_block_factor(blockFactor);
    tilingData.set_ub_factor(ubFactor);
    tilingData.set_num_row(numRow);
    tilingData.set_num_col(numCol);
    tilingData.set_avg_factor(avgFactor);
    return true;
}

void SetDefaultTiling()
{
    tilingData.set_is_gemma(1);
    tilingData.set_row_factor(FLOAT_PER_REAPEAT);
    tilingData.set_reduce_mask(0);
    tilingData.set_last_reduce_mask(0);
    tilingData.set_rstd_size(0);
    tilingData.set_left_num(0);
    tilingData.set_last_left_num(0);
}

void setWorkSize(gert::TilingContext *context)
{
    size_t usrSize = 256;
    size_t sysWorkspaceSize = SYS_WORKSPACE_SIZE;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
}

static ge::graphStatus Tiling4GemmaRmsNorm(gert::TilingContext *context)
{
    if (!CheckInputShape4RmsNorm(context)) {
        std::printf("CheckInputShape4RmsNorm failed, Input shape invalid.");
        return ge::GRAPH_FAILED;
    }

    auto xDesc = context->GetInputDesc(0);
    auto xDataType = xDesc->GetDataType();

    uint32_t numCore;
    uint32_t dataPerBlock;
    uint64_t ubSize;
    setPlateCoreInfo(context, numCore, ubSize);
    ubSize = ubSize - UB_USED;

    uint32_t xDtypeKey = DTYPE_KEY_FP16;
    SetByDtype(xDataType, xDtypeKey, dataPerBlock);
    if (!getUbFactor(dataPerBlock, xDtypeKey, numCore, context)) {
        std::printf("Tiling4GemmaRmsNorm failed,check getUbFactor.");
        return ge::GRAPH_FAILED;
    }

    SetDefaultTiling();
    setWorkSize(context);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4RmsNorm(gert::TilingParseContext *context)
{
    auto compileInfo = GetCompileInfoPtr<Tiling4RmsNormCompileInfo>(context);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RmsNorm).Tiling(Tiling4RmsNorm).TilingParse<Tiling4RmsNormCompileInfo>(TilingPrepare4RmsNorm);

}  // namespace optiling

namespace ops {

static ge::graphStatus InferShape4RmsNorm(gert::InferShapeContext *context)
{
    // get input shapes
    const gert::Shape *x_shape = context->GetInputShape(0);
    const gert::Shape *gamma_shape = context->GetInputShape(1);
    // get output shapes
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;

    size_t xDimNum = x_shape->GetDimNum();
    size_t gammaDimNum = gamma_shape->GetDimNum();

    gert::Shape *rstd_shape = context->GetOutputShape(1);
    rstd_shape->SetDimNum(xDimNum);
    for (size_t i = 0; i < xDimNum; i++) {
        if (i < xDimNum - gammaDimNum) {
            rstd_shape->SetDim(i, x_shape->GetDim(i));
        } else {
            rstd_shape->SetDim(i, 1);
        }
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4RmsNorm(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNorm).InferShape(InferShape4RmsNorm).InferDataType(InferDataType4RmsNorm);

class RmsNorm : public OpDef {
public:
    explicit RmsNorm(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

    }
};
OP_ADD(RmsNorm);
}  // namespace ops