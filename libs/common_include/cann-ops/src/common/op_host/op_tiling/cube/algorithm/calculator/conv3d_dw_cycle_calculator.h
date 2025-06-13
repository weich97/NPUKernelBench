/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
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
 * \file conv3d_dw_cycle_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_DW_CALCULATOR_CYCLE_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_DW_CALCULATOR_CYCLE_CALCULATOR_H_

#include "cube/algorithm/calculator/cycle_calculator.h"

namespace optiling {
namespace cachetiling {
class Conv3DDwCycleCalculator : public CycleCalculator {
 public:
  Conv3DDwCycleCalculator(SingleCoreStatus &core_status);
  ~Conv3DDwCycleCalculator() override = default;
  bool Init(const CubeTilingParam &params) override;

 protected:
  void GenBlockDimsMapFactors() override;
  bool IsValidBatchDim(int32_t batch_dim) const override;
  void UpdateTilingShape() override;
  bool LoopBlockDims(bool prune, int32_t used_core) override;

 private:
  std::vector<int32_t> dout_dims_factors_;
  std::vector<int32_t> kd_dims_factors_;

  int32_t d_dim_ = 1;
  int32_t kd_dim_ = 1;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_DW_CALCULATOR_CYCLE_CALCULATOR_H_