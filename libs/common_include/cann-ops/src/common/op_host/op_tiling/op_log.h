/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file op_log.h
 * \brief
 */
#ifndef OPS_COMMON_INC_OP_LOG_H_
#define OPS_COMMON_INC_OP_LOG_H_
#define unlikely(x) __builtin_expect((x), 0)
#define OP_LOGI(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
#define OP_LOGE(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
#define OP_LOGD(op_name, ...) std::printf(op_name, ##__VA_ARGS__)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do                                                                                                           \
  {                                                                                                            \
    if (unlikely(condition))                                                                                   \
    {                                                                                                          \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)            \
    if ((ptr) == nullptr)                                    \
    {                                                        \
        std::printf("nullptr error!");                       \
        return ge::GRAPH_SUCCESS;                            \
    }  

#define OP_LOGD_FULL(opname, ...) OP_LOG_FULL(DLOG_DEBUG, get_op_info(opname), __VA_ARGS__)


#define OP_LOGI_IF_RETURN(condition, return_value, op_name, fmt, ...)                                          \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGI(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

#define OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)

#endif  // OPS_COMMON_INC_OP_LOG_H_
