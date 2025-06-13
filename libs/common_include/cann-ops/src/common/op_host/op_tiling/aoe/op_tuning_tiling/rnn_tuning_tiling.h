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
#ifndef RUNTIME_KB_TUNING_TILING_RNN_TUNING_TILING_H_
#define RUNTIME_KB_TUNING_TILING_RNN_TUNING_TILING_H_

#include "register/tuning_tiling_registry.h"
#include "register/tuning_bank_key_registry.h"
namespace tuningtiling {
struct DynamicRnnInputArgsV2 {
  int64_t timeStep;
  int64_t batchSize;
  int64_t inputSize;
  int64_t hiddenSize;
  bool biasFlag;
  ge::DataType xType;
  ge::DataType wType;
  ge::DataType bType;
  ge::Format xformat;
  ge::Format wformat;
  ge::Format bformat;
};
BEGIN_TUNING_TILING_DEF(MatMulTunnerTiling)
TUNING_TILING_DATA_FIELD_DEF(uint32_t, batchDim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, mDim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, nDim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, kDim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseM);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseN);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseK);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, dbAFlag);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, dbBFlag);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, dbCFlag);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, iterateOrder);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(MatMulTunnerTiling, FIELD(MatMulTunnerTiling, batchDim), FIELD(MatMulTunnerTiling, mDim),
               FIELD(MatMulTunnerTiling, nDim), FIELD(MatMulTunnerTiling, kDim), FIELD(MatMulTunnerTiling, baseM),
               FIELD(MatMulTunnerTiling, baseN), FIELD(MatMulTunnerTiling, baseK), FIELD(MatMulTunnerTiling, dbAFlag),
               FIELD(MatMulTunnerTiling, dbBFlag), FIELD(MatMulTunnerTiling, dbCFlag),
               FIELD(MatMulTunnerTiling, iterateOrder));

BEGIN_TUNING_TILING_DEF(DynamicRnnTunnerTiling)
TUNING_TILING_DATA_FIELD_DEF(uint32_t, scheduleId);
TUNING_TILING_DATA_FIELD_DEF(MatMulTunnerTiling, mmTiling);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(DynamicRnnTunnerTiling, FIELD(DynamicRnnTunnerTiling, scheduleId),
               FIELD(DynamicRnnTunnerTiling, mmTiling));
}  // namespace tuningtiling
#endif
