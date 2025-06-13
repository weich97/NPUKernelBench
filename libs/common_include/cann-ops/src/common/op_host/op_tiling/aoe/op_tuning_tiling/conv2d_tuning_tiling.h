/**
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef CONV2D_TUNING_TILING_H_
#define CONV2D_TUNING_TILING_H_ 
#include "register/tuning_bank_key_registry.h"
#include "register/tuning_tiling_registry.h"

namespace tuningtiling {
#pragma pack(push, 1)
struct Conv2DInputArgs {
int64_t batch;
int64_t fmci;
int64_t ciOri;
int64_t hi;
int64_t wi;
int64_t n;
int64_t wci;
int64_t kh;
int64_t kw;
int64_t ho;
int64_t wo;
int64_t padu;
int64_t padd;
int64_t padl;
int64_t padr;
int64_t dilationsH;
int64_t dilationsW;
int64_t strideH;
int64_t strideW;
int64_t groups;
int64_t groupsOri;
float preFusionUbUtilize;
float postFusionUbUtilize;
float preFusionUbEltwiseNx1;
float postFusionUbEltwiseNx1;
float preFusionUbBroadcast;
float postFusionUbBroadcast;
float preFusionUbBroadcastNx1;
float postFusionUbBroadcastNx1;
float postFusionUbChannelwise;
int64_t preFusionVectorUtilize;
int64_t postFusionVectorUtilize;
int64_t hf32Mode;
bool biasFlag;
bool offsetDescFlag;
bool broadcastFlag;
ge::DataType aType;
ge::DataType bType;
ge::DataType cType;
ge::DataType madType;
ge::DataType biasType;
ge::DataType scaleType;
ge::DataType quantConv2dCubType;
ge::Format inputFormat;
ge::Format weightFormat;
ge::Format outputFormat;
};
#pragma pack(pop)

BEGIN_TUNING_TILING_DEF(Conv2DTunnerTiling)
TUNING_TILING_DATA_FIELD_DEF(uint32_t, batchSingleCore);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, nSingleCore);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, batchDim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, nDim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, mDim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, groupDim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, cubN);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, nUbL0cFactor);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, mL0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, kL0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, mAl1Factor);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, nBl1Factor);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, kAl116);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, kBl116);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, kAl1Factor);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, kBl1Factor);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, kAub);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, mAub);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, allPbBitValue);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, pbAL1Value);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, pbBL1Value);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, pbAL0Value);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, pbBL0Value);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, pbCL0Value);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, pbAUBValue);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, pbBUBValue);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, pbCUBValue);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, pbUBGValue);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, tilingId);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(Conv2DTunnerTiling,
    FIELD(Conv2DTunnerTiling, batchSingleCore),
    FIELD(Conv2DTunnerTiling, nSingleCore),
    FIELD(Conv2DTunnerTiling, batchDim),
    FIELD(Conv2DTunnerTiling, nDim),
    FIELD(Conv2DTunnerTiling, mDim),
    FIELD(Conv2DTunnerTiling, groupDim),
    FIELD(Conv2DTunnerTiling, cubN),
    FIELD(Conv2DTunnerTiling, nUbL0cFactor),
    FIELD(Conv2DTunnerTiling, mL0),
    FIELD(Conv2DTunnerTiling, kL0),
    FIELD(Conv2DTunnerTiling, mAl1Factor),
    FIELD(Conv2DTunnerTiling, nBl1Factor),
    FIELD(Conv2DTunnerTiling, kAl116),
    FIELD(Conv2DTunnerTiling, kBl116),
    FIELD(Conv2DTunnerTiling, kAl1Factor),
    FIELD(Conv2DTunnerTiling, kBl1Factor),
    FIELD(Conv2DTunnerTiling, kAub),
    FIELD(Conv2DTunnerTiling, mAub),
    FIELD(Conv2DTunnerTiling, allPbBitValue),
    FIELD(Conv2DTunnerTiling, pbAL1Value),
    FIELD(Conv2DTunnerTiling, pbBL1Value),
    FIELD(Conv2DTunnerTiling, pbAL0Value),
    FIELD(Conv2DTunnerTiling, pbBL0Value),
    FIELD(Conv2DTunnerTiling, pbCL0Value),
    FIELD(Conv2DTunnerTiling, pbAUBValue),
    FIELD(Conv2DTunnerTiling, pbBUBValue),
    FIELD(Conv2DTunnerTiling, pbCUBValue),
    FIELD(Conv2DTunnerTiling, pbUBGValue),
    FIELD(Conv2DTunnerTiling, blockNum),
    FIELD(Conv2DTunnerTiling, tilingId)
    );
}  // namespace tuningtiling
 
 #endif