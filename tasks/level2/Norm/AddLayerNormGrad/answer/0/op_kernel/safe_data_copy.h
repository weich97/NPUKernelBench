/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file safe_data_copy.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_ASCENDC_SAFE_DATA_COPY_H_
#define OPS_BUILT_IN_OP_ASCENDC_SAFE_DATA_COPY_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "platform.h"

template <bool forAtomicAdd = false, typename T>
__aicore__ inline void SafeDataCopy(const AscendC::GlobalTensor<T> &dstGlobal, const AscendC::LocalTensor<T> &srcLocal,
    const int64_t &calCount, bool recoverUbTailFormat = false)
{
    constexpr int typeSize = sizeof(T); // Implementation note.
    constexpr int numElemsPerBlock = AscendC::ONE_BLK_SIZE / typeSize; // Implementation note.
    if constexpr (PlatformSocInfo::IsDataCopyPadSupport() &&
                  sizeof(T) < 8) { // Implementation note.
        AscendC::DataCopyParams copyParams{1, static_cast<uint16_t>(calCount * typeSize), 0, 0};
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    } else {
        if (likely(!(calCount % numElemsPerBlock))) { // Implementation note.
            struct AscendC::DataCopyParams copyParams;
            copyParams.blockLen = calCount / AscendC::AscendCUtils::GetC0Count(typeSize);
            DataCopy(dstGlobal, srcLocal, copyParams);
        } else { // Implementation note.
            const int numAlignedBlocks = calCount / numElemsPerBlock * numElemsPerBlock; // Implementation note.
            if (calCount * typeSize < AscendC::ONE_BLK_SIZE) {
                DataCopy(dstGlobal, srcLocal, numElemsPerBlock);
                return; // Implementation note.
            }
            DataCopy(dstGlobal, srcLocal, numAlignedBlocks);
            event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventID);
            const int rollbackEleCount = calCount - numAlignedBlocks; // Implementation note.
            const size_t rollbackDstIdx = numAlignedBlocks - numElemsPerBlock; // Implementation note.
            const size_t rollbackSrcIdx = rollbackDstIdx + rollbackEleCount; // Implementation note.
            if constexpr (!forAtomicAdd) {
                for (int i = 0; i < numElemsPerBlock;
                     ++i) { // Implementation note.
                    srcLocal.SetValue((rollbackDstIdx + i), srcLocal.GetValue(rollbackSrcIdx + i)); // Implementation note.
                }
            } else {
                const size_t setZeroEleCount = numElemsPerBlock - rollbackEleCount; // Implementation note.
                for (int i = 0; i < setZeroEleCount; ++i) {
                    srcLocal.SetValue((rollbackDstIdx + i), 0); // Implementation note.
                }
                for (int i = setZeroEleCount; i < numElemsPerBlock; ++i) { // Implementation note.
                    srcLocal.SetValue((rollbackDstIdx + i), srcLocal.GetValue(rollbackSrcIdx + i)); // Implementation note.
                }
                DataCopy(dstGlobal[calCount - numElemsPerBlock], srcLocal[rollbackDstIdx], numElemsPerBlock);
                return; // Implementation note.
            }
            DataCopy(dstGlobal[calCount - numElemsPerBlock], srcLocal[rollbackDstIdx], numElemsPerBlock);
            if (recoverUbTailFormat) { // Implementation note.
                event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
                DataCopy(
                    srcLocal[rollbackDstIdx], dstGlobal[rollbackDstIdx], numElemsPerBlock); // Implementation note.
            }
        }
    }
}

#endif  // OPS_BUILT_IN_OP_ASCENDC_SAFE_DATA_COPY_H_