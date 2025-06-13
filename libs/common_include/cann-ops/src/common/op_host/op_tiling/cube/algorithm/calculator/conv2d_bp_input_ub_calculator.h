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
 * \file conv2d_bp_input_ub_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_UB_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_UB_CALCULATOR_H_

#include "cube/algorithm/calculator/calculator.h"

namespace optiling {
namespace cachetiling {
struct UbParams {
  int32_t n_cub;
  int32_t db_cub;
  int32_t *k_al1_factors;
  size_t idx_k;
};

class Conv2DBpInputUbCalculator : public Calculator {
 public:
  Conv2DBpInputUbCalculator(SingleCoreStatus &core_status, int32_t &h_num);
  ~Conv2DBpInputUbCalculator() override = default;
  bool Exec() override;
  int32_t GetWoAub() const { return wo_aub_; }
  bool Init(const CubeTilingParam &params) override;
  void Clear();

 private:
  void UpdateSinleCoreStatus();
  bool CalcUbStatus();
  bool GetBestUbFactors(int64_t &max_dma_size);
  bool InitUbDb(const int64_t &wo_aub, int64_t &max_dma_size);
  void GetAubSize(int64_t &max_dma_size, bool &first_flag, UbParams &ub_params);
  int32_t GetAubM(const int32_t &aub_size, const int32_t &k_aub);
  int32_t GetWoAub(const int32_t &aub_size, const int32_t &k_aub, const int32_t &aub_m, const int32_t &m_size);

  const Conv2DBpInputTilingParam *dx_param_ = nullptr;
  const Shape *dedy_ = nullptr;
  const Shape *kernel_ = nullptr;
  const Shape *dedx_ = nullptr;
  UbStatus ub_status_;
  int32_t wo_aub_ = 1;
  const L0Status &l0_status_;
  const L1Status &l1_status_;
  const int32_t &h_num_;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_CONV2D_BP_INPUT_UB_CALCULATOR_H_