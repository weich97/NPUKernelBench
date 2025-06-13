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
 * \file dw_cache_tiling_impl.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_DW_CACHE_TILING_IMPL_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3D_DW_CACHE_TILING_IMPL_H_

#include "cube/algorithm/dw_cache_tiling_impl.h"

namespace optiling {
namespace cachetiling {
class Conv3DDwCacheTilingImpl : public DwCacheTilingImpl {
 public:
  explicit Conv3DDwCacheTilingImpl(const CubeTilingParam &params);
  virtual ~Conv3DDwCacheTilingImpl() = default;
  bool Init(const CubeTilingParam &params) override;

 protected:
  void SetOrigShape() override;
  void SetTiling(CubeTiling &tiling) const override;
  bool CheckCycleModelUnsupport() const override;
  int32_t CalcTilingId(const CubeTiling &tiling, const TilingIdParam &id_param) const override;
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_DW_CACHE_TILING_IMPL_H_