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
#ifndef OP_TUNING_TILING_TUNE_SPACE_LOG_H
#define OP_TUNING_TILING_TUNE_SPACE_LOG_H

#include <cstdint>
#include <memory>
#include "slog.h"
#include "mmpa_api.h"

namespace OpTuneSpace {
using Status = uint32_t;
constexpr Status SUCCESS = 0;
constexpr Status FAILED = 1;
constexpr Status TILING_NUMBER_EXCEED = 2;

constexpr int TUNE_MODULE = static_cast<int>(TUNE);
#define TUNE_SPACE_LOGD(format, ...) \
do {DlogSub(TUNE_MODULE, "TUNE_SPACE", DLOG_DEBUG, "[Tid:%d]" #format"\n", mmGetTid(), ##__VA_ARGS__);} while (0)

#define TUNE_SPACE_LOGI(format, ...) \
    do {DlogSub(TUNE_MODULE, "TUNE_SPACE", DLOG_INFO, "[Tid:%d]" #format"\n", mmGetTid(), ##__VA_ARGS__);} while(0)

#define TUNE_SPACE_LOGW(format, ...) \
    do {DlogSub(TUNE_MODULE, "TUNE_SPACE", DLOG_WARN, "[Tid:%d]" #format"\n", mmGetTid(), ##__VA_ARGS__);} while(0)

#define TUNE_SPACE_LOGE(format, ...) \
    do {DlogSub(TUNE_MODULE, "TUNE_SPACE", DLOG_ERROR, "[Tid:%d]" #format"\n", mmGetTid(), ##__VA_ARGS__);} while(0)

#define TUNE_SPACE_LOGV(format, ...) \
    do {DlogSub(TUNE_MODULE, "TUNE_SPACE", DLOG_EVENT, "[Tid:%d]" #format"\n", mmGetTid(), ##__VA_ARGS__);} while(0)

#define TUNE_SPACE_MAKE_SHARED(execExpr0, execExpr1) \
    do {                                            \
        try {                                       \
            (execExpr0);                            \
        } catch (const std::bad_alloc &) {          \
            TUNE_SPACE_LOGE("Make shared failed");    \
            execExpr1;                              \
        }                                           \
    } while (false)
} // namespace OpTuneSpace
#endif