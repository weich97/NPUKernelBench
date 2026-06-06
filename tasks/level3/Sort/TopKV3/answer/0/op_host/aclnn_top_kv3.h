/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_TOP_K_V3_H_
#define OP_API_INC_TOP_K_V3_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Implementation note.
 * @domain aclnn_ops_train
 *
 * Implementation note.
 *
 * @param [in] self: npu
 * Implementation note.
 * Implementation note.
 * @param [in] k:
 * Implementation note.
 * @param [in] dim:
 * Implementation note.
 * @param [in] largest:
 * Implementation note.
 * @param [in] sorted:
 * Implementation note.
 * Implementation note.
 * @param [in] valuesOut:
 * Implementation note.
 * Implementation note.
 * @param [in] indicesOut:
 * Implementation note.
 * Implementation note.
 * Implementation note.
 * Implementation note.
 */
ACLNN_API aclnnStatus aclnnTopKV3GetWorkspaceSize(const aclTensor* self, int64_t k, int64_t dim, bool largest,
                                                bool sorted, aclTensor* valuesOut, aclTensor* indicesOut,
                                                uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * Implementation note.
 *
 * Implementation note.
 *
 * Implementation note.
 * Implementation note.
 * Implementation note.
 * Implementation note.
 * Implementation note.
 */
ACLNN_API aclnnStatus aclnnTopKV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TOP_K_V3_H_
