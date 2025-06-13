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
 * \file ub_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_UB_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_UB_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
class UbCalculator : public Calculator {
 public:
  UbCalculator(SingleCoreStatus &core_status);
  virtual ~UbCalculator() = default;
  bool Exec() override;
  bool Init(const CubeTilingParam &params) override;
  int32_t GetTilingWiBub() const;

 private:
  void CalcUbStatus();
  void UpdateSinleCoreStatus();
  int32_t CalcWiBub(int32_t limit_hn, int32_t k_bub) const;
  void CalcKBub(int32_t limit_hn, int32_t bl1_hi);

  UbStatus ub_status_;
  int32_t wi_bub_ = 0;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_UB_CALCULATOR_H_