#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 1u;

template <typename uIdxType, typename idxType, typename dataType>
class KernelCoalesceSparse {
public:
    __aicore__  inline KernelCoalesceSparse() = default;
    __aicore__  inline void Init(GM_ADDR uniqueIndices, GM_ADDR indices, GM_ADDR values, GM_ADDR newIndices,
                                 GM_ADDR newValue, const CoalesceSparseTilingData* __restrict tilingData);
    __aicore__  inline void Process();

private:
    __aicore__  inline void InitTilingValue(const CoalesceSparseTilingData* __restrict tilingData);
    __aicore__  inline void CopyIn(uint64_t repeatTime, uint64_t moveLen);
    __aicore__  inline void ComputeAndCopyOut(uint64_t repeatTime, uint64_t taskLen);
    __aicore__ inline void valueMove(uint64_t destGmOffset, uint64_t srcGmOffset, uint64_t moveValueLen);
    __aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> uniqueIndicesQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> indicesQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> valueQueue;

    GlobalTensor<uIdxType> uniqueIndicesGm;
    GlobalTensor<idxType> indicesGm;
    GlobalTensor<dataType> valueGm;

    GlobalTensor<idxType> newIndicesGm;
    GlobalTensor<dataType> newValueGm;

    uint64_t usedCoreNum {0};
    uint64_t m {0};
    uint64_t valueSize {0};
    uint64_t taskNum {0};
    uint64_t taskTail {0};
    uint64_t moveOneSize {0};
    uint64_t taskRepeatTimes {0};
    uint64_t taskRepeatTail {0};
    uint64_t taskTailRepeatTimes {0};
    uint64_t taskTailRepeatTail {0};
    uint64_t moveValueTimes {0};
    uint64_t moveValueLen {0};
    uint64_t moveValueTail {0};

    uint64_t blockSize {32};
    uint64_t taskLen {0};
    uint64_t coreTaskTimes {0};
    uint64_t coreTaskTail {0};
    uint64_t unquieBlockPreSize {0};
    uint64_t idBlockPreSize {0};
    uint32_t indicesUbStride {0};
    uint64_t mByte {0};
    uint64_t indicesAlign32 {0};
    const CoalesceSparseTilingData* __restrict tilingDevice {nullptr};
};

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::Init(
    GM_ADDR uniqueIndices, GM_ADDR indices, GM_ADDR values, GM_ADDR newIndices, GM_ADDR newValue,
    const CoalesceSparseTilingData* __restrict tilingData) {
    InitTilingValue(tilingData);
    uint64_t coreId = GetBlockIdx();
    uint64_t beginOffset = coreId * taskNum;
    uint64_t indicesBeginOffset = beginOffset * m;
    uint64_t valueBeginOffset = beginOffset * valueSize;

    if (coreId < usedCoreNum - 1) {
        taskLen = taskNum;
        coreTaskTimes = taskRepeatTimes;
        coreTaskTail = taskRepeatTail;
    }
    else if (coreId == usedCoreNum - 1) {
        taskLen = taskTail;
        coreTaskTimes = taskTailRepeatTimes;
        coreTaskTail = taskTailRepeatTail;
    }
    //SetGlobalBuffer
    this->uniqueIndicesGm.SetGlobalBuffer((__gm__ uIdxType *)uniqueIndices + beginOffset, taskLen);
    this->indicesGm.SetGlobalBuffer((__gm__ idxType *)indices + indicesBeginOffset, taskLen * m);
    this->valueGm.SetGlobalBuffer((__gm__ dataType *)values + valueBeginOffset, taskLen * valueSize);
    this->newIndicesGm.SetGlobalBuffer((__gm__ idxType *)newIndices);
    this->newValueGm.SetGlobalBuffer((__gm__ dataType *)newValue);
    indicesUbStride = (uint32_t)(CeilDiv(m * sizeof(idxType), blockSize));
    // //block_pre_size
    unquieBlockPreSize = blockSize  / sizeof(uIdxType);
    idBlockPreSize = blockSize  / sizeof(idxType);
    indicesAlign32 = indicesUbStride * idBlockPreSize;
    // //InitBuffer Need Align32
    this->pipe.InitBuffer(this->uniqueIndicesQueue, BUFFER_NUM, moveOneSize * sizeof(uIdxType));
    this->pipe.InitBuffer(this->indicesQueue, BUFFER_NUM, moveOneSize * indicesUbStride * blockSize);
    // moveValueLen is align 32
    uint64_t moveValueLenAlign32 = CeilDiv(moveValueLen * sizeof(dataType), blockSize) * blockSize;
    this->pipe.InitBuffer(this->valueQueue, BUFFER_NUM, moveValueLenAlign32);
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::InitTilingValue(const CoalesceSparseTilingData* __restrict tilingData) {
    //Get tilingData
    this->tilingDevice = tilingData;
    usedCoreNum = tilingDevice->usedCoreNum;
    m = tilingDevice->m;
    mByte = m * sizeof(idxType);
    valueSize = tilingDevice->valueSize;
    taskNum = tilingDevice->taskNum;
    taskTail = tilingDevice->taskTail;
    moveOneSize = tilingDevice->moveOneSize;
    taskRepeatTimes = tilingDevice->taskRepeatTimes;
    taskRepeatTail = tilingDevice->taskRepeatTail;
    taskTailRepeatTimes = tilingDevice->taskTailRepeatTimes;
    taskTailRepeatTail = tilingDevice->taskTailRepeatTail;
    moveValueTimes = tilingDevice->moveValueTimes;
    moveValueLen = tilingDevice->moveValueLen;
    moveValueTail = tilingDevice->moveValueTail;
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::Process() {
    // taskAlign32
    for (uint64_t i = 0; i < coreTaskTimes; i++){
        CopyIn(i, moveOneSize);
        ComputeAndCopyOut(i, moveOneSize);
    }
    // taskTail
    if (coreTaskTail != 0){
        CopyIn(coreTaskTimes, coreTaskTail);
        ComputeAndCopyOut(coreTaskTimes, coreTaskTail);
    }
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::CopyIn(uint64_t repeatTime, uint64_t moveLen) {

    uint64_t taksOffset = repeatTime * moveOneSize;
    uint64_t uniqueIndicesOffset = taksOffset;
    uint64_t indicesOffset = taksOffset * m;

    DataCopyExtParams copyParams_indices {(uint16_t)moveLen, (uint32_t)(mByte), 0, 0, 0};
    DataCopyPadExtParams<idxType> indices_padParams{true, 0, 0, 0};

    LocalTensor<uIdxType> uniqueIndicesLocal = uniqueIndicesQueue.AllocTensor<uIdxType>();
    LocalTensor<idxType> indicesLocal = indicesQueue.AllocTensor<idxType>();

    uint64_t unquieIndicesAlign32 = CeilDiv(moveLen, unquieBlockPreSize) * unquieBlockPreSize;

    DataCopy(uniqueIndicesLocal, uniqueIndicesGm[uniqueIndicesOffset], unquieIndicesAlign32);
    DataCopyPad(indicesLocal, indicesGm[indicesOffset], copyParams_indices, indices_padParams);
    uniqueIndicesQueue.EnQue<uIdxType>(uniqueIndicesLocal);
    indicesQueue.EnQue<idxType>(indicesLocal);
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::ComputeAndCopyOut(uint64_t repeatTime, uint64_t taskLen) {
    LocalTensor<uIdxType> uniqueIndicesLocal = uniqueIndicesQueue.DeQue<uIdxType>();
    LocalTensor<idxType> indicesLocal = indicesQueue.DeQue<idxType>();
     for(uint64_t i = 0; i < taskLen; i++) {
        // This is the DESTINATION index in the new coalesced tensor.
        int64_t uniqueIndicesId = uniqueIndicesLocal.GetValue(i);
        
        // This is the SOURCE index within the block of data this core is responsible for.
        uint64_t src_element_idx_in_chunk = repeatTime * moveOneSize + i;

        // **FIX**: Calculate DESTINATION offsets for both new_indices and new_values.
        // NOTE: The original index copy logic is inefficient but we focus on the value bug first.
        int64_t gmIndicesOffset = uniqueIndicesId * m;
        int64_t dest_gm_offset_base = uniqueIndicesId * valueSize;
        DataCopyParams copyParams_indices{1, (uint16_t)(mByte), 0, 0};
        DataCopyPad(newIndicesGm[gmIndicesOffset], indicesLocal[i * indicesAlign32], copyParams_indices);

        // **FIX**: Calculate the SOURCE offset for the original values tensor.
        uint64_t src_gm_offset_base = src_element_idx_in_chunk * valueSize;

        for (int j = 0; j < moveValueTimes; j++) {
            event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
            uint64_t value_chunk_offset = j * moveValueLen;
            // Pass the correct DESTINATION and SOURCE offsets to valueMove
            valueMove(dest_gm_offset_base + value_chunk_offset, src_gm_offset_base + value_chunk_offset, moveValueLen);
        }
        if (moveValueTail > 0) {
            event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
            uint64_t value_chunk_offset = moveValueTimes * moveValueLen;
            // Pass the correct DESTINATION and SOURCE offsets for the tail
            valueMove(dest_gm_offset_base + value_chunk_offset, src_gm_offset_base + value_chunk_offset, moveValueTail);
        }
    }
    uniqueIndicesQueue.FreeTensor(uniqueIndicesLocal);
    indicesQueue.FreeTensor(indicesLocal);
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline void KernelCoalesceSparse<uIdxType, idxType, dataType>::valueMove(uint64_t destGmOffset, uint64_t srcGmOffset, uint64_t moveValueLen) {
    uint64_t valueByte = moveValueLen * sizeof(dataType);
    LocalTensor<dataType> valueLocal = valueQueue.AllocTensor<dataType>();
    DataCopyExtParams copyParams_value_ {(uint16_t)1, (uint32_t)(valueByte) , 0, 0, 0};
    DataCopyPadExtParams<dataType> values_padParams{true, 0, 0, 0};
    
    // **FIX**: Use the correct srcGmOffset to read from the input Global Memory
    DataCopyPad(valueLocal, valueGm[srcGmOffset], copyParams_value_, values_padParams);
    
    event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
    DataCopyParams copyParams_value{1, (uint16_t)(valueByte), 0, 0};
    SetAtomicAdd<dataType>();

    // **FIX**: Use the correct destGmOffset to write to the output Global Memory
    DataCopyPad(newValueGm[destGmOffset], valueLocal, copyParams_value);

    SetAtomicNone();
    valueQueue.FreeTensor(valueLocal);
}

template <typename uIdxType, typename idxType, typename dataType>
__aicore__ inline uint64_t KernelCoalesceSparse<uIdxType, idxType, dataType>::CeilDiv(uint64_t x, uint64_t y) {
    return y == 0 ? x : (x + y - 1) / y;
}

extern "C" __global__ __aicore__ void coalesce_sparse(GM_ADDR unique_len, GM_ADDR unique_indices, GM_ADDR indices,
                                                      GM_ADDR values, GM_ADDR new_indices, GM_ADDR new_value,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const CoalesceSparseTilingData* __restrict tilingDevice = &tilingData;
    if (TILING_KEY_IS(0)) {
        KernelCoalesceSparse<int64_t, int64_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(1)) {
        KernelCoalesceSparse<int64_t, int64_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(2)) {
        KernelCoalesceSparse<int64_t, int64_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(3)) {
        KernelCoalesceSparse<int64_t, int32_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(4)) {
        KernelCoalesceSparse<int64_t, int32_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(5)) {
        KernelCoalesceSparse<int64_t, int32_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(6)) {
        KernelCoalesceSparse<int32_t, int64_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(7)) {
        KernelCoalesceSparse<int32_t, int64_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(8)) {
        KernelCoalesceSparse<int32_t, int64_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(9)) {
        KernelCoalesceSparse<int32_t, int32_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(10)) {
        KernelCoalesceSparse<int32_t, int32_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(11)) {
        KernelCoalesceSparse<int32_t, int32_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }
}