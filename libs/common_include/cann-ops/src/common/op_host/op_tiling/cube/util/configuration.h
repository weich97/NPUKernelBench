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
 * \file configuration.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_UTIL_CONFIGURATION_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_UTIL_CONFIGURATION_H_

namespace optiling {
namespace cachetiling {
class Configuration {
 public:
  static Configuration &Instance();
  static bool IsDebugMode();

 private:
  Configuration();
  ~Configuration() {};
  bool is_debug_mode_ = false;
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_UTIL_CONFIGURATION_H_