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
 * \file shape.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_ENTITY_SHAPE_H
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_ENTITY_SHAPE_H

#include <cstdint>
#include <string>

namespace optiling {
namespace cachetiling {
class TilingShape {
 public:
  std::string ToString() const;
  void Init();

  int64_t batch = 1;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t group = 1;
  int64_t h = 1;
  int64_t w = 1;
  int64_t din = 1;
  int64_t dk = 1;
  int64_t dout = 1;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_ENTITY_SHAPE_H