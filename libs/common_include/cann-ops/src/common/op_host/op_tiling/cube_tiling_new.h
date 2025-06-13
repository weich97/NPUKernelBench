/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
 * \file cube_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_TILING_NEW_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_TILING_NEW_H_

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// #include "external/graph/operator.h"
#include "op_tiling.h"

namespace optiling {
const size_t kConv2dDimNumLimit = 4;
const int32_t kConv2dNDim = 0;
const int32_t kConv2dHDim = 2;
const int32_t kConv2dWDim = 3;

/*
 * @brief: tiling function of cube category operators
 * @param [in] curShape: execution time shape info
 * @param [in] opInfo: compile time generated info of operator
 * @param [out] runInfo: result data
 * @return int: tiling id
 */
bool cube_tiling(const std::string &op_type, const std::vector<int64_t> &input_shape,
                 const std::vector<int64_t> &var_value);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_TILING_H_