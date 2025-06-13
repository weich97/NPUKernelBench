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
 * \file configuration.cc
 * \brief
 */

#include "cube/util/configuration.h"

#include <cstdint>
#include "op_log.h"

namespace optiling {
namespace cachetiling {
Configuration &Configuration::Instance() {
  static Configuration inst;
  return inst;
}

Configuration::Configuration() {
  int32_t enable = 1; //CheckLogLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
  is_debug_mode_ = (enable == 1);
}

bool Configuration::IsDebugMode() { return Instance().is_debug_mode_; }
}  // namespace cachetiling
}  // namespace optiling