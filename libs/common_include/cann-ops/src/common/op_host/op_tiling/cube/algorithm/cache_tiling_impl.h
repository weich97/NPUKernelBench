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
 * \file cache_tiling_impl.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CACHE_TILING_IMPL_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CACHE_TILING_IMPL_H_

#include "cube/algorithm/entity/status.h"
#include "cube/include/cache_tiling.h"
#include "cube/include/cube_tiling.h"
#include "cube/include/cube_tiling_param.h"
#include "cube/util/registry.h"

namespace optiling {
namespace cachetiling {
class CacheTilingImpl {
 public:
  explicit CacheTilingImpl(const CubeTilingParam &params);
  virtual ~CacheTilingImpl() { params_ = nullptr; }
  virtual bool GenTiling(CubeTiling &tiling) = 0;
  virtual bool Init(const CubeTilingParam &params);
  virtual void Clear();

 protected:
  void UpdateTiling(CubeTiling &tiling) const;
  void UpdateTilingBlockDims(CubeTiling &tiling) const;
  void UpdateTilingL1Status(CubeTiling &tiling) const;
  void UpdateTilingL0Status(CubeTiling &tiling) const;
  void UpdateTilingUbStatus(CubeTiling &tiling) const;

  const CubeTilingParam *params_ = nullptr;
  SingleCoreStatus single_core_status_;
};

DECLARE_REGISTRY_TYPE(CacheTilingFactory, CacheTilingImpl, const CubeTilingParam &);
#define REGISTER_TILING_GENERATOR(op_type, derived_clazz) \
  REGISTER_TYPE_CLASS(CacheTilingFactory, op_type, derived_clazz)
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CACHE_TILING_IMPL_H_