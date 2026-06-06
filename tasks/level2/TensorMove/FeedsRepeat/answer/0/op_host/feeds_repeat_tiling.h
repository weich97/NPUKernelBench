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
 * \file feeds_repeat_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_FEEDS_REPEAT_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_FEEDS_REPEAT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FeedsRepeatTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, length); // Implementation note.
    TILING_DATA_FIELD_DEF(uint32_t, length_aligned); // Implementation note.
    TILING_DATA_FIELD_DEF(int64_t, elem_row); // Implementation note.
    TILING_DATA_FIELD_DEF(int64_t, elem_per_loop); // Implementation note.
    TILING_DATA_FIELD_DEF(int64_t, max_core_num); // Implementation note.
    TILING_DATA_FIELD_DEF(int64_t, core_per_group); // Implementation note.
    TILING_DATA_FIELD_DEF(int64_t, core_moreover); // Implementation note.
    TILING_DATA_FIELD_DEF(int64_t, empty_size); // Implementation note.
    TILING_DATA_FIELD_DEF(int64_t, row_per_core); // Implementation note.
    TILING_DATA_FIELD_DEF(int64_t, row_left); // Implementation note.
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FeedsRepeat, FeedsRepeatTilingData)
}  // namespace optiling

struct FeedsRepeatCompileInfo {
    uint64_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
};

#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_FEEDS_REPEAT_TILING_H