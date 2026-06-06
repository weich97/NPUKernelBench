/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_MSE_LOSS_BACKWARD_H_
#define OP_API_INC_MSE_LOSS_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Implementation note.
 * @domain aclnn_ops_train
 *
 * Implementation note.
 *
 * @param [in] gradOutput：npu
 * Implementation note.
 * Implementation note.
 * Implementation note.
 * Implementation note.
 * @param [in] target：npu
 * Implementation note.
 * Implementation note.
 * Implementation note.
 * Implementation note.
 * Implementation note.
 * Implementation note.
 * Implementation note.
 * Implementation note.
 * Implementation note.
 */
ACLNN_API aclnnStatus aclnnMseLossBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                           const aclTensor* target, int64_t reduction, aclTensor* out,
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
ACLNN_API aclnnStatus aclnnMseLossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MSE_LOSS_BACKWARD_H_
