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

#ifndef TUNE_SPACE_MATMUL_TUNE_SPACE_H
#define TUNE_SPACE_MATMUL_TUNE_SPACE_H

#include "aoe/op_tuning_tiling/gemm_tuning_tiling.h"

#include "tune_space.h"
#include "tune_space_register.h"

namespace OpTuneSpace {

class MatMulTuneSpace : public TuneSpace {
public:
    explicit MatMulTuneSpace() = default;
    ~MatMulTuneSpace() override = default;

    Status GetTuneSpace(gert::TilingContext* op, std::vector<nlohmann::json> &jsonTuneSpace) override;
    virtual std::string GetOpType() = 0;
};

class BatchMatMulV2TuneSpace : public MatMulTuneSpace {
private:
    std::string GetOpType() override { return "BatchMatMulV2"; };
};

class MatMulV2TuneSpace : public MatMulTuneSpace {
private:
    std::string GetOpType() override { return "MatMulV2"; };
};

}
#endif // namespace OpTuneSpace