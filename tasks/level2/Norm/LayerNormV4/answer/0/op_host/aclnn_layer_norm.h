/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_LAYER_NORM_H_
#define OP_API_INC_LEVEL2_LAYER_NORM_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Implementation note.
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnLayerNormGetWorkspaceSize(const aclTensor *input,
    const aclIntArray *normalizedShape, const aclTensor *weightOptional, const aclTensor *biasOptional, double eps,
    aclTensor *out, aclTensor *meanOutOptional, aclTensor *rstdOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * Implementation note.
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnLayerNormWithImplModeGetWorkspaceSize(const aclTensor *input,
    const aclIntArray *normalizedShape, const aclTensor *weightOptional, const aclTensor *biasOptional, double eps,
    aclTensor *out, aclTensor *meanOutOptional, aclTensor *rstdOutOptional, int32_t implMode, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * Implementation note.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnLayerNorm(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

/**
 * Implementation note.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnLayerNormWithImplMode(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_LAYER_NORM_H_
