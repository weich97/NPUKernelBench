/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef OP_TUNING_TILING_TUNE_SPACE_H
#define OP_TUNING_TILING_TUNE_SPACE_H

#include <vector>
#include <nlohmann/json.hpp>

#include "exe_graph/runtime/tiling_context.h"
#include "common_utils.h"
#include "tune_space_log.h"

namespace OpTuneSpace {

/**
 * class analyze input tiling space
*/
class TuneSpace {
public:
    explicit TuneSpace(){};
    virtual ~TuneSpace(){};

    /**
     * analyze and split tiling space
     * param [in] args     input tiling args
     * return     spaces   tiling space
    */
   virtual Status GetTuneSpace(gert::TilingContext* op, std::vector<nlohmann::json> &jsonTuneSpace) = 0;
};
} // namespace OpTuneSpace
#endif // OP_TUNING_TILING_TUNE_SPACE_H_
