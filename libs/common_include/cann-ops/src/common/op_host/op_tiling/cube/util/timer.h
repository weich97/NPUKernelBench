/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

/*!
 * \file timer.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_UTIL_TIMER_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_UTIL_TIMER_H_

#include <chrono>
#include <cstdint>

#include "op_tiling.h"

namespace optiling {
namespace cachetiling {
#define OP_EVENT(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
// class Timer {
//  public:
//   Timer(const char *op_type, const char *desc);
//   ~Timer();
//   void Start();
//   void End();
//   int64_t Elapsed() const;

//  private:
//   const char *op_type_ = nullptr;
//   const char *desc_ = nullptr;
//   std::chrono::time_point<std::chrono::steady_clock> start_time_;
//   std::chrono::time_point<std::chrono::steady_clock> end_time_;
// };

#define CACHE_TILING_TIME_STAMP_START(name)                          \
  std::chrono::time_point<std::chrono::steady_clock> __start_##name; \
  if (::optiling::prof_switch) {                                     \
    __start_##name = std::chrono::steady_clock::now();               \
  }

#define CACHE_TILING_TIME_STAMP_END(name, op_type)                                                          \
  if (::optiling::prof_switch) {                                                                            \
    std::chrono::time_point<std::chrono::steady_clock> __end_##name = std::chrono::steady_clock::now();     \
    OP_EVENT(op_type, "[TILING PROF]" #name "cost time: %ld us",                                            \
             std::chrono::duration_cast<std::chrono::microseconds>(__end_##name - __start_##name).count()); \
  }
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_UTIL_TIMER_H_