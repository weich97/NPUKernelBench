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
 * \file calculator.cc
 * \brief
 */
#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
bool Calculator::IsL1SizeValid(int64_t l1_size) const {
  return params_->platform_info.IsValidL1Size(l1_size);
}

bool Calculator::Init(const CubeTilingParam &params) {
  params_ = &params;
  return true;
}

void Calculator::Clear() {
  params_ = nullptr;
}
}  // namespace cachetiling
}  // namespace optiling