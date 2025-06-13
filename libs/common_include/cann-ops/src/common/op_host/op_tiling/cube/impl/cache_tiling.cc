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
 * \file cache_tiling.cc
 * \brief
 */
#include "cube/include/cache_tiling.h"

#include "cube/algorithm/cache_tiling_impl.h"
// #include "cube/util/timer.h"
#include "op_log.h"

namespace optiling {
namespace cachetiling {


static FactoryInst<std::shared_ptr<CacheTilingImpl>> inst_;

bool GenTiling(const CubeTilingParam &params, CubeTiling &tiling) {
  // CACHE_TILING_TIME_STAMP_START(GenTiling);
  // OP_LOG_FULL(DLOG_DEBUG, params.op_type, "[CubeTilingParam][%s]", params.ToString().c_str());
  if (!params.IsValid()) {
    OP_LOGE(params.op_type, "Invalid input param");
    return false;
  }

  auto impl = inst_.Get(params.type);
  if (impl == nullptr) {
    impl = CacheTilingFactory().Create(params.type, params);
    if (impl == nullptr) {
      OP_LOGE(params.op_type, "Creator TilingGenerator failed");
      return false;
    }
    inst_.Add(params.type, impl);
  }
  if (!impl->Init(params)) {
    OP_LOGE(params.op_type, "Failed to init TilingImpl!");
    return false;
  }
  bool res = impl->GenTiling(tiling);
  impl->Clear();
  // CACHE_TILING_TIME_STAMP_END(GenTiling, params.op_type);
  if (!tiling.IsValid()) {
    OP_LOGE(params.op_type, "Invalid output param");
    return false;
  }
  return res;
  // return false;
}

void DestoryTilingFactory() {
  // inst_.Clear();
}
}  // namespace cachetiling
}  // namespace optiling
