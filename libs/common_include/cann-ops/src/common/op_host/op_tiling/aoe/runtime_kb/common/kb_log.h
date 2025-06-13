/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNTIME_KB_COMMON_UTILS_KB_LOG_H_
#define RUNTIME_KB_COMMON_UTILS_KB_LOG_H_
#include "slog.h"
#include "mmpa_api.h"

constexpr int TUNE_MODULE = static_cast<int>(TUNE);

#define CANNKB_LOGD(format, ...)                                                                   \
  do {                                                                                             \
    DlogSub(TUNE_MODULE, "CANNKB", DLOG_DEBUG, "[Tid:%d]" format "\n", mmGetTid(), ##__VA_ARGS__); \
  } while (0)

#define CANNKB_LOGI(format, ...)                                                                  \
  do {                                                                                            \
    DlogSub(TUNE_MODULE, "CANNKB", DLOG_INFO, "[Tid:%d]" format "\n", mmGetTid(), ##__VA_ARGS__); \
  } while (0)

#define CANNKB_LOGW(format, ...)                                                                  \
  do {                                                                                            \
    DlogSub(TUNE_MODULE, "CANNKB", DLOG_WARN, "[Tid:%d]" format "\n", mmGetTid(), ##__VA_ARGS__); \
  } while (0)

#define CANNKB_LOGE(format, ...)                                                                   \
  do {                                                                                             \
    DlogSub(TUNE_MODULE, "CANNKB", DLOG_ERROR, "[Tid:%d]" format "\n", mmGetTid(), ##__VA_ARGS__); \
  } while (0)

#define CANNKB_LOGEVENT(format, ...)                                                               \
  do {                                                                                             \
    DlogSub(TUNE_MODULE, "CANNKB", DLOG_EVENT, "[Tid:%d]" format "\n", mmGetTid(), ##__VA_ARGS__); \
  } while (0)
#endif
