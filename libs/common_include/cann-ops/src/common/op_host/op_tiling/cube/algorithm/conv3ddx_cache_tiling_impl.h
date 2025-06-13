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
 * \file conv3ddx_cache_tiling_impl.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3DDX_CACHE_TILING_IMPL_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3DDX_CACHE_TILING_IMPL_H_

#include "cube/algorithm/cache_tiling_impl.h"
#include "cube/algorithm/calculator/conv3d_bp_input_tiling_calculator.h"

namespace optiling {
namespace cachetiling {

class Conv3DDxCacheTilingImpl : public CacheTilingImpl {
 public:
  explicit Conv3DDxCacheTilingImpl(const CubeTilingParam &params);
  ~Conv3DDxCacheTilingImpl() override = default;
  bool GenTiling(CubeTiling &tiling) override;
  bool Init(const CubeTilingParam &params) override;
  void Clear() override;

 private:
  void ShowResourceStatistics() const;
  void SetOrigShape();
  void SetTiling(Conv3DBpInputTiling &tiling) const;

  const Conv3DBpInputTilingParam *conv3ddx_params_ = nullptr;
  Conv3DBpInputTilingCalculator conv3ddx_tiling_calculator_;
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_Conv3DDX_CACHE_TILING_IMPL_H_
