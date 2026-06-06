/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_top_kv3.h"
#include "top_kv3_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/small_vector.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

const int64_t MAX_AICORE_CALC_INPUTSIZE = 32768;
const int64_t MAX_AICORE_CALC_DIM = 8;
const int64_t MAX_SUPPORT_DIMS_NUMS = 8;

static const std::initializer_list<op::DataType> ASCEND910A_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  }
  return ASCEND910A_DTYPE_SUPPORT_LIST;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *values, const aclTensor *indices) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(values, return false);
  OP_CHECK_NULL(indices, return false);
  return true;
}

static int64_t MakeWrapDim(int64_t dim, int64_t dimPostExpr) {
  // Implementation note.
  if (dimPostExpr <= 0) {
    dimPostExpr = 1;
  }
  if (dim < 0) {
    dim += dimPostExpr;
  }
  return dim;
}

static bool CheckParamValid(const aclTensor *self, int64_t k, int64_t dim) {
  // Implementation note.
  auto inputShape = self->GetViewShape();
  int64_t tmpDim = static_cast<int64_t>(inputShape.GetDimNum());
  if (tmpDim == 0 && dim != 0 && dim != -1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [-1, 0], but got %ld)",
            dim);
    return false;
  } else if (tmpDim > 0 && (dim < -tmpDim || dim >= tmpDim)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [-%ld, %ld],"
            "but got %ld)", tmpDim, tmpDim - 1, dim);
    return false;
  }

  // Implementation note.
  int64_t positiveDim = MakeWrapDim(dim, tmpDim);
  int64_t tmpK = (tmpDim > 0) ? inputShape.GetDim(positiveDim) : 1;
  if (k < 0 || k > tmpK) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Selected index k out of range (max num of self.size(%ld) is %ld,"
            "but k is %ld)", tmpDim, tmpK, k);
    return false;
  }
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *values, const aclTensor *indices) {
  auto supportList = GetDtypeSupportList();
  // Implementation note.
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_MATCH(values, self->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(indices, op::DataType::DT_INT64, return false);
  return true;
}

static bool CheckShape(const aclTensor *self) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t k, int64_t dim,
                               const aclTensor *values, const aclTensor *indices) {
  // Implementation note.
  CHECK_RET(CheckNotNull(self, values, indices), ACLNN_ERR_PARAM_NULLPTR);

  // Implementation note.
  CHECK_RET(CheckParamValid(self, k, dim), ACLNN_ERR_PARAM_INVALID);

  // Implementation note.
  CHECK_RET(CheckDtypeValid(self, values, indices), ACLNN_ERR_PARAM_INVALID);

  // Implementation note.
  CHECK_RET(CheckShape(self), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static const aclTensor *TopkAdaptInputZeroDimTensor(const aclTensor *self, int64_t dimNum, aclOpExecutor *executor) {
  if (dimNum != 0) {
    return self;
  }
  int64_t selfShapeValue[1] = {1};
  aclIntArray *selfShape = executor->AllocIntArray(selfShapeValue, 1);
  auto selfReshape = l0op::Reshape(self, selfShape, executor);
  return selfReshape;
}

static bool CheckCalcInAiCore(const aclTensor *self, int64_t k) {
  auto inputShape = self->GetViewShape();
  int64_t tmpDim = static_cast<int64_t>(inputShape.GetDimNum());
  int64_t inputSize = 1;
  for (int64_t i = 0; i < tmpDim; i++) {
    inputSize *= inputShape.GetDim(i);
  }
  if (inputSize > MAX_AICORE_CALC_INPUTSIZE) {
    return k >= MAX_AICORE_CALC_DIM;
  }
  return true;
}

static const aclTensor *TopkAdaptGeCastTensor(const aclTensor *self, const aclTensor *value, int64_t k,
                                              op::DataType dataType, aclOpExecutor *executor) {
  SocVersion version = GetCurrentPlatformInfo().GetSocVersion();
  if (version == SocVersion::ASCEND910 && CheckCalcInAiCore(self, k) && self->GetDataType() == op::DataType::DT_FLOAT) {
    return l0op::Cast(value, dataType, executor);
  }
  return value;
}

static aclIntArray *GetDimTransposeArray(int64_t dimNum, int64_t lastDim, int64_t positiveDim,
                                         aclOpExecutor *executor) {
  std::vector<int64_t> perm(dimNum, 0);
  for (int64_t i = 0; i < dimNum; i++) {
    perm[i] = i;
  }
  std::swap(perm[positiveDim], perm[lastDim]);
  return executor->AllocIntArray(perm.data(), dimNum);
}

aclnnStatus aclnnTopKV3GetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool largest, bool sorted,
                                      aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize,
                                      aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnTopKV3, DFX_IN(self, k, dim, largest, sorted), DFX_OUT(valuesOut, indicesOut));

   // Implementation note.
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // Implementation note.
  auto ret = CheckParams(self, k, dim, valuesOut, indicesOut);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // Implementation note.
  if (self->IsEmpty() || valuesOut->IsEmpty() || indicesOut->IsEmpty()) {
    // Implementation note.
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // Implementation note.
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  int64_t dimNum = static_cast<int64_t>(selfContiguous->GetViewShape().GetDimNum());
  int64_t positiveDim = MakeWrapDim(dim, dimNum);
  int64_t lastDim = MakeWrapDim(-1, dimNum);

  auto selfReshape = TopkAdaptInputZeroDimTensor(selfContiguous, dimNum, uniqueExecutor.get());
  CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  aclTensor *indicesCastInt32 = nullptr;
  aclTensor *valuesTopkOut = nullptr;

  // Implementation note.
  auto selfCast = TopkAdaptGeCastTensor(selfReshape, selfReshape, k, op::DataType::DT_FLOAT16, uniqueExecutor.get());
  CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (positiveDim != lastDim) {
    aclIntArray *axes = GetDimTransposeArray(dimNum, lastDim, positiveDim, uniqueExecutor.get());
    CHECK_RET(axes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Implementation note.
    auto selfTranspose = l0op::Transpose(selfCast, axes, uniqueExecutor.get());
    CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Implementation note.
    auto topkOut = l0op::Topk(selfTranspose, k, lastDim, largest, sorted, uniqueExecutor.get());
    valuesTopkOut = std::get<0>(topkOut);
    aclTensor *indicesTopkOut = std::get<1>(topkOut);
    CHECK_RET(valuesTopkOut != nullptr && indicesTopkOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Implementation note.
    valuesTopkOut = const_cast<aclTensor *>(l0op::Transpose(valuesTopkOut, axes, uniqueExecutor.get()));

    // Implementation note.
    indicesCastInt32 = const_cast<aclTensor *>(l0op::Transpose(indicesTopkOut, axes, uniqueExecutor.get()));
  } else {
    auto topkOut = l0op::Topk(selfCast, k, positiveDim, largest, sorted, uniqueExecutor.get());
    valuesTopkOut = std::get<0>(topkOut);
    indicesCastInt32 = std::get<1>(topkOut);
  }
  CHECK_RET(valuesTopkOut != nullptr && indicesCastInt32!= nullptr, ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(CheckReduceOutShape(valuesOut, valuesTopkOut), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckReduceOutShape(indicesOut, indicesCastInt32), ACLNN_ERR_PARAM_INVALID);

  // Implementation note.
  auto valuesCast = TopkAdaptGeCastTensor(selfReshape, valuesTopkOut, k, op::DataType::DT_FLOAT, uniqueExecutor.get());
  CHECK_RET(valuesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // Implementation note.
  auto viewCopyValuesResult = l0op::ViewCopy(valuesCast, valuesOut, uniqueExecutor.get());
  CHECK_RET(viewCopyValuesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // Implementation note.
  auto indicesCastInt64 = l0op::Cast(indicesCastInt32, op::DataType::DT_INT64, uniqueExecutor.get());
  CHECK_RET(indicesCastInt64 != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // Implementation note.
  auto viewCopyIndicesResult = l0op::ViewCopy(indicesCastInt64, indicesOut, uniqueExecutor.get());
  CHECK_RET(viewCopyIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // Implementation note.
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTopKV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnTopKV3);
  // Implementation note.
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
