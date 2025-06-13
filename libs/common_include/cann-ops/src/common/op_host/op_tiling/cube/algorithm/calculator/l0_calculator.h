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
 * \file l0_calculator.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_L0_CALCULATOR_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_L0_CALCULATOR_H_

#include <string>
#include <array>

#include "cube/algorithm/calculator/calculator.h"
#include "cube/algorithm/entity/shape.h"

namespace optiling {
namespace cachetiling {
namespace {
constexpr size_t kL0FactorMaxSize = 4;

class CalcStatus {
 public:
  std::string ToString() const;

  int32_t db_l0a = 1;
  int32_t db_l0b = 1;
  int32_t db_l0c = 1;
  int32_t max_mk = 1;
  int32_t max_nk = 1;
  int32_t max_mn = 1;
  int32_t max_axis_num = 1;
  int32_t max_axis_pnt = 1;
};
}  // namespace

class L0Calculator : public Calculator {
 public:
  L0Calculator(SingleCoreStatus &core_status);
  virtual ~L0Calculator() = default;
  bool Exec() override;
  bool Init(const CubeTilingParam &params) override;

 private:
  CalcStatus InitCalcStatus(int32_t db_l0c) const;
  void GenL0Factors(const CalcStatus &calc_status, int32_t l0c_db_type);
  void CalcL0Factors(L0Status &l0_status, L1Status &l1_status, int32_t k0_max, int32_t l0c_db_type, bool even_k_factor);
  void CalcL0Status(const DimFactor &factor, int32_t l0c_db_type);
  int64_t CalcLoadSize() const;
  int64_t FullLoadSize() const;
  int64_t KFullLoadSize() const;
  int64_t SingleKFullLoadSize(KLoadType load_type, int64_t k_full_load_size, int32_t min_load_size) const;
  int64_t NeitherFullLoadSize() const;
  void UpdateSinleCoreStatus();
  bool NeedUpdate(int64_t load_size, int64_t scope, int64_t l0c_used) const;
  void Update(int64_t load_size, int64_t scope, int64_t l0c_used);

  const TilingShape &shape_;
  L0Status l0_status_;
  L1Status l1_status_;

  int64_t extend_shape_k_ = 0;
  int64_t load_size_ = INT64_MAX;
  int64_t l0c_used_ = 0;
  int64_t scope_ = 0;
  L0Status best_l0_status_;
};
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CALCULATOR_L0_CALCULATOR_H_