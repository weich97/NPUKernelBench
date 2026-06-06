/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file isamin_aiv.h
 */
#ifndef ISAMIN_H
#define ISAMIN_H

#include <type_traits>
#include <cfloat>
#include "kernel_operator.h"

namespace Isamin {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_BLOCK = 32;
constexpr int32_t BYTES_PER_REPEAT = 256;
constexpr int32_t MAX_REPEATS = 255;
constexpr int32_t BLOCKS_PER_REPEAT = 8;
constexpr int32_t MAX_CAST_COUNT = 512;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t MAX_VECTOT = 40;
constexpr uint64_t REDUCE_MAX_RST_MASK = 72340172838076673;  // 2^0+2^8+2^16+2^24+2^32+2^40+2^48+2^56

// Implementation note.
constexpr uint64_t REDUCE_MAX_CORES_RST_MASK = 6148914691236517205;  // (i*2)^0+(i*2)^2+(i*2)^4+..+(i*2)^31

constexpr int32_t GM_RESULT_LEN = 2;

constexpr uint32_t MAX_NUM_F32_ELE_EACH_CORE = 23040;

// Implementation note.
// Implementation note.
// Implementation note.
constexpr int32_t DEAL_TIMES_EACH_CORE_REDUCE = 63;
constexpr int32_t DEAL_TIMES_EACH_CORE_REDUCE_REP = 8; // Implementation note.

template <typename T>
class Isamin {
public:
    __aicore__ inline Isamin<T>(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR usrWorkspace, GM_ADDR tiling);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };
    __aicore__ inline bool ParseTilingData(GM_ADDR tiling);
    __aicore__ inline void CopyIn(uint32_t calCount, uint32_t offset);
    __aicore__ inline void CopyInAttacheTail(uint32_t calCount, uint32_t tailCount, uint32_t offset);
    __aicore__ inline void SingleProcess(const int32_t count, uint32_t k);
    __aicore__ inline void GetReduceMinCount(uint32_t k);
    __aicore__ inline void reduceMinTmpResult(uint32_t k);
    __aicore__ inline void getCoreTmpReduResult();
    __aicore__ inline void CopyTmpRstToWkGM();
    __aicore__ inline void reduceMinCoresResult();
    __aicore__ inline void copyInCoresTmpRst();
    __aicore__ inline void ReduceMinCoresTmpRst();
    __aicore__ inline void copyOutRst();

    __aicore__ inline void getCoreReduResult();
    __aicore__ inline void CopyTmpRstOut();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inDataQueue;
    TBuf<QuePosition::VECCALC> reduceMinRsts;
    TBuf<QuePosition::VECCALC> reduceMinWrk;
    TBuf<QuePosition::VECCALC> reduceMinDst;
    TBuf<QuePosition::VECCALC> rMinRsts;

    // Implementation note.
    TQue<QuePosition::VECOUT, BUFFER_NUM> outDataQueue;

    TQue<QuePosition::VECIN, 1> coresRstInDataQueue;
    TBuf<QuePosition::VECCALC> coresRstReduceMinWrk;
    TQue<QuePosition::VECOUT, 1> coresRstOutDataQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> tmpRstWkGM;
    GlobalTensor<int32_t> outTensorsGM;

    LocalTensor<T> rMinALLRstsTenor;
    LocalTensor<T> rMinTmpRstsTenor;
    LocalTensor<T> rMinTmpWrkTenor;
    LocalTensor<T> rMinDstTenor;

    LocalTensor<T> coresRstReduceTmp;
    LocalTensor<int32_t> rMinRstsTenor;

    uint32_t blockIdx = 0;
    uint32_t elementsPerBlock = BYTE_BLOCK / BYTE_LEN_4;
    uint32_t elementsPerRepeat = BYTES_PER_REPEAT / BYTE_LEN_4;

    uint32_t incx;
    uint32_t needVecCoreNum;
    uint32_t dytpeFlag;
    uint32_t startOffset;
    uint32_t eleTotalEachCore;

    uint32_t dealTimesEachCore;
    uint32_t dealLenEachTime;
    uint32_t reduceMinRstsLen;
    uint32_t dealLenUpBlockEachTime;
    uint32_t coresRstReduceMinLen;
    uint32_t rstLenAllCoreBytes;
    uint32_t tailCount;
    uint32_t tailEle;

    uint32_t maxRepeatLen;
    uint32_t totalRptCntNor;
    uint32_t totalRptCntNorRemainder;  // should calc
    uint32_t rptBatchCntNor;           // limit by L0 API, should calc
    uint32_t rptBatchCntNorRemainder;  // should calc
    uint32_t rmdRptLenNor;
    bool isUpdateRptParas = false;

    uint32_t startElement;
    uint32_t dealEleEachTime;
};

template <typename T>
__aicore__ inline void Isamin<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR usrWorkspace, GM_ADDR tiling)
{
    this->blockIdx = GetBlockIdx();

    if (false == ParseTilingData(tiling)) {
        return;
    }

    inTensorsGM.SetGlobalBuffer((__gm__ T *)x + startOffset, eleTotalEachCore);
    tmpRstWkGM.SetGlobalBuffer((__gm__ T *)usrWorkspace + blockIdx * GM_RESULT_LEN, GM_RESULT_LEN);
    outTensorsGM.SetGlobalBuffer((__gm__ int32_t *)y, GM_RESULT_LEN);

    this->pipe.InitBuffer(this->inDataQueue, BUFFER_NUM, this->dealLenUpBlockEachTime * BYTE_LEN_4);
    this->pipe.InitBuffer(this->reduceMinRsts,
                          this->reduceMinRstsLen * BYTE_LEN_4); // Implementation note.
    this->pipe.InitBuffer(this->reduceMinWrk, this->reduceMinRstsLen * BYTE_LEN_4);
    this->pipe.InitBuffer(this->reduceMinDst, this->elementsPerBlock * BYTE_LEN_4);
    this->pipe.InitBuffer(this->rMinRsts, this->elementsPerBlock * BYTE_LEN_4);
    this->pipe.InitBuffer(this->outDataQueue, BUFFER_NUM, this->elementsPerBlock * BYTE_LEN_4);

    // Implementation note.
    // Implementation note.
    this->coresRstReduceMinLen = this->elementsPerRepeat * 2;
    this->pipe.InitBuffer(this->coresRstInDataQueue, 1, this->coresRstReduceMinLen * BYTE_LEN_4);
    this->pipe.InitBuffer(this->coresRstReduceMinWrk, this->coresRstReduceMinLen * BYTE_LEN_4);
    this->pipe.InitBuffer(this->coresRstOutDataQueue, 1, this->elementsPerBlock * BYTE_LEN_4);

    // Implementation note.
    this->rMinALLRstsTenor = reduceMinRsts.Get<T>();
    this->rMinTmpRstsTenor = this->rMinALLRstsTenor[this->elementsPerBlock];
    this->rMinTmpWrkTenor = reduceMinWrk.Get<T>();
    this->rMinDstTenor = reduceMinDst.Get<T>();
    this->rMinRstsTenor = rMinRsts.Get<int32_t>();
    this->coresRstReduceTmp = coresRstReduceMinWrk.Get<T>();
}

template <typename T>
__aicore__ inline bool Isamin<T>::ParseTilingData(GM_ADDR tiling)
{
    auto tilingBuf = reinterpret_cast<__gm__ uint32_t *>(tiling);
    this->incx = (*(__gm__ uint32_t *)(tilingBuf + 0));
    this->needVecCoreNum = (*(__gm__ uint32_t *)(tilingBuf + 1));

    // Implementation note.
    if (this->blockIdx + 1 > this->needVecCoreNum) {
        return false;
    }

    this->dytpeFlag = (*(__gm__ uint32_t *)(tilingBuf + 2));
    this->rstLenAllCoreBytes = (*(__gm__ uint32_t *)(tilingBuf + 3));
    this->tailCount = (*(__gm__ uint32_t *)(tilingBuf + 4));
    this->maxRepeatLen = (*(__gm__ uint32_t *)(tilingBuf + 5));

    this->startOffset = (*(__gm__ uint32_t *)(tilingBuf + 6 + this->blockIdx));
    this->eleTotalEachCore = (*(__gm__ uint32_t *)(tilingBuf + 6 + 1 * MAX_VECTOT + this->blockIdx));
    this->dealTimesEachCore = (*(__gm__ uint32_t *)(tilingBuf + 6 + 2 * MAX_VECTOT + this->blockIdx));
    this->dealLenEachTime = (*(__gm__ uint32_t *)(tilingBuf + 6 + 3 * MAX_VECTOT + this->blockIdx));
    this->reduceMinRstsLen = (*(__gm__ uint32_t *)(tilingBuf + 6 + 4 * MAX_VECTOT + this->blockIdx));
    this->dealLenUpBlockEachTime = (*(__gm__ uint32_t *)(tilingBuf + 6 + 5 * MAX_VECTOT + this->blockIdx));

    this->totalRptCntNor = (*(__gm__ uint32_t *)(tilingBuf + 6 + 6 * MAX_VECTOT + this->blockIdx));
    this->totalRptCntNorRemainder = (*(__gm__ uint32_t *)(tilingBuf + 6 + 7 * MAX_VECTOT + this->blockIdx));
    this->rptBatchCntNor = (*(__gm__ uint32_t *)(tilingBuf + 6 + 8 * MAX_VECTOT + this->blockIdx));
    this->rptBatchCntNorRemainder = (*(__gm__ uint32_t *)(tilingBuf + 6 + 9 * MAX_VECTOT + this->blockIdx));
    this->rmdRptLenNor = (*(__gm__ uint32_t *)(tilingBuf + 6 + 10 * MAX_VECTOT + this->blockIdx));

    this->startElement = this->startOffset;
    this->dealEleEachTime = this->dealLenEachTime;
    this->tailEle = this->tailCount;
    if (this->dytpeFlag == 1) { // Implementation note.
        this->startElement = this->startOffset / 2;
        this->dealEleEachTime = this->dealEleEachTime / 2;
        this->tailEle = this->tailCount / 2;
    }

    return true;
}

template <typename T>
__aicore__ inline void Isamin<T>::Process()
{
    // Implementation note.
    if (this->blockIdx + 1 > this->needVecCoreNum) {
        SyncAll();
        return;
    }
    Duplicate<T>(this->rMinALLRstsTenor, FLT_MAX, this->reduceMinRstsLen);

    uint32_t calCount = 0;
    uint32_t copyedlCount = 0;
    int32_t leftCount = 0;
    // Implementation note.
    int32_t dealBigTimes = this->dealTimesEachCore / DEAL_TIMES_EACH_CORE_REDUCE;
    int32_t dealBigTimesTail = this->dealTimesEachCore % DEAL_TIMES_EACH_CORE_REDUCE;

    // Implementation note.
    for (uint32_t k = 0; k < this->dealTimesEachCore; k++) {
        // Implementation note.
        calCount = this->dealLenEachTime;
        this->isUpdateRptParas = false;
        leftCount = this->eleTotalEachCore - copyedlCount;

        if (leftCount < this->dealLenEachTime) {
            calCount = leftCount;
            isUpdateRptParas = true;
        }

        // Implementation note.
        if (this->blockIdx == 0 && this->tailCount > 0 && k == 0) {
            CopyInAttacheTail(calCount, this->tailCount, copyedlCount);
            calCount += this->tailCount;
            isUpdateRptParas = true;
        } else {
            CopyIn(calCount, copyedlCount);
        }

        SingleProcess(calCount, k);
        copyedlCount += calCount;

        // Implementation note.
        uint32_t dealedTimes = k + 1;
        if ((dealedTimes % DEAL_TIMES_EACH_CORE_REDUCE == 0 && k > 0) || (k == this->dealTimesEachCore - 1)) {
            reduceMinTmpResult(k);
            // Implementation note.
            if ((0 != dealBigTimesTail) && (dealedTimes / DEAL_TIMES_EACH_CORE_REDUCE == dealBigTimes) &&
                dealedTimes != this->dealTimesEachCore) {
                Duplicate<T>(this->rMinTmpRstsTenor, FLT_MAX, this->reduceMinRstsLen - this->elementsPerBlock);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    if (this->needVecCoreNum == 1 && this->blockIdx == 0) {
        // Implementation note.
        SyncAll();
        getCoreReduResult();
        CopyTmpRstOut();
    } else {
        getCoreTmpReduResult();
        CopyTmpRstToWkGM();
        SyncAll();
        if (this->blockIdx == 0) {
            reduceMinCoresResult();
        }
    }
}

template <typename T>
__aicore__ inline void Isamin<T>::CopyIn(uint32_t calCount, uint32_t offset)
{
    LocalTensor<T> inLocalTensor = inDataQueue.AllocTensor<T>();
    DataCopy(inLocalTensor, inTensorsGM[offset], calCount);
    inDataQueue.EnQue(inLocalTensor);
}

template <typename T>
__aicore__ inline void Isamin<T>::CopyInAttacheTail(uint32_t calCount, uint32_t tailCount, uint32_t offset)
{
    LocalTensor<T> inLocalTensor = inDataQueue.AllocTensor<T>();

    if (calCount > 0) {
        DataCopy(inLocalTensor, inTensorsGM[offset], calCount);
    }

    // Implementation note.
    uint16_t blockCout = (uint16_t)1;
    uint16_t blockLen = tailCount * BYTE_LEN_4;
    DataCopyParams copyParams{blockCout, blockLen, 0, 0};
    DataCopyPadParams padParams{true, 0, 0, 0}; // Implementation note.
    DataCopyPad(inLocalTensor[calCount], inTensorsGM[offset + calCount], copyParams, padParams);

    inDataQueue.EnQue(inLocalTensor);
}

template <typename T>
__aicore__ inline void Isamin<T>::SingleProcess(const int32_t count, uint32_t k)
{
    LocalTensor<T> srcLocal = inDataQueue.DeQue<T>();

    // Implementation note.
    if (k % DEAL_TIMES_EACH_CORE_REDUCE != 0 && k > 0) { // Implementation note.
        GetReduceMinCount(k - 1);
    }

    uint32_t totalRepeatCnt = this->totalRptCntNor;
    uint32_t totalRepeatCntRemainder = this->totalRptCntNorRemainder;  // should calc
    uint32_t repeatBatchCnt = this->rptBatchCntNor;                    // limit by L0 API, should calc
    uint32_t repeatBatchCntRemainder = this->rptBatchCntNorRemainder;  // should calc
    uint32_t maxRepeatLen = this->maxRepeatLen;
    uint32_t rmdRepeatLen = this->rmdRptLenNor;
    if (this->isUpdateRptParas == true) {
        totalRepeatCnt = count / elementsPerRepeat;
        totalRepeatCntRemainder = count % elementsPerRepeat;     // should calc
        repeatBatchCnt = totalRepeatCnt / MAX_REPEATS;           // limit by L0 API, should calc
        repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS;  // should calc
        rmdRepeatLen = repeatBatchCntRemainder * elementsPerRepeat;
    }

    uint32_t offset = 0;

    for (uint32_t i = 0; i < repeatBatchCnt; i++) {
        Abs(srcLocal[offset], srcLocal[offset], elementsPerRepeat, MAX_REPEATS, {1, 1, 8, 8});
        offset += maxRepeatLen;
    }

    if (repeatBatchCntRemainder > 0) {
        Abs(srcLocal[offset], srcLocal[offset], elementsPerRepeat, repeatBatchCntRemainder, {1, 1, 8, 8});
        offset += rmdRepeatLen;
    }

    if (totalRepeatCntRemainder > 0) {
        Abs(srcLocal[offset], srcLocal[offset], totalRepeatCntRemainder, 1, {1, 1, 8, 8});
    }

    AscendC::PipeBarrier<PIPE_V>();

    int32_t ReduceMinCount = count;
    // Implementation note.
    if (this->dytpeFlag == 1) {
        ReduceMinCount = count / 2;
        offset = 0;
        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            PairReduceSum(srcLocal[offset / 2], srcLocal[offset], MAX_REPEATS, elementsPerRepeat, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            offset += maxRepeatLen;
        }

        if (repeatBatchCntRemainder > 0) {
            PairReduceSum(srcLocal[offset / 2], srcLocal[offset], repeatBatchCntRemainder, elementsPerRepeat, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            offset += rmdRepeatLen;
        }

        if (totalRepeatCntRemainder > 0) {
            PairReduceSum(srcLocal[offset / 2], srcLocal[offset], 1, totalRepeatCntRemainder, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    AscendC::SetMaskCount();
    AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, ReduceMinCount);
    WholeReduceMin<T, false>(srcLocal, srcLocal, AscendC::MASK_PLACEHOLDER, 1, 1, 1, 8);
    AscendC::SetMaskNorm();
    AscendC::ResetMask();

    // Implementation note.
    if (((k + 1) % DEAL_TIMES_EACH_CORE_REDUCE == 0 && k > 0) || k == this->dealTimesEachCore - 1) {
        GetReduceMinCount(k);
    }

    inDataQueue.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void Isamin<T>::GetReduceMinCount(uint32_t k)
{
    uint32_t tmpK = k % DEAL_TIMES_EACH_CORE_REDUCE * this->elementsPerBlock;

    // Implementation note.
    event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID2);
    WaitFlag<HardEvent::V_S>(eventID2);

    float val = 0; // Implementation note.
    float idx = 0; // Implementation note.
    GetReduceMaxMinCount<T>(val, idx); // Implementation note.

    this->rMinTmpRstsTenor.SetValue(tmpK, val);
    this->rMinTmpRstsTenor.SetValue(tmpK + 1, idx);
}

template <typename T>
__aicore__ inline void Isamin<T>::reduceMinTmpResult(uint32_t k)
{
    AscendC::PipeBarrier<PIPE_V>();

    uint64_t mask[2] = {REDUCE_MAX_RST_MASK, 0};

    ReduceMin(this->rMinDstTenor, this->rMinALLRstsTenor, this->rMinTmpWrkTenor, mask, DEAL_TIMES_EACH_CORE_REDUCE_REP,
              8, true);

    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID1);
    WaitFlag<HardEvent::V_S>(eventID1);

    float tmpK = this->rMinDstTenor.GetValue(1); // Implementation note.
    int32_t maxIdxInRst = *reinterpret_cast<uint32_t *>(&tmpK);  //

    float tmpK2 = this->rMinALLRstsTenor.GetValue(maxIdxInRst + 1);
    int32_t maxIdxInOriVec = *reinterpret_cast<uint32_t *>(&tmpK2);

    // Implementation note.
    if (maxIdxInRst != 0) { // Implementation note.
        int32_t preOffset = (k / DEAL_TIMES_EACH_CORE_REDUCE) * DEAL_TIMES_EACH_CORE_REDUCE;
        int32_t maxValueIndex = this->startElement +
                                (preOffset + maxIdxInRst / this->elementsPerBlock - 1) * this->dealEleEachTime +
                                maxIdxInOriVec;
        // Implementation note.
        // Implementation note.
        if (this->tailEle != 0 && this->blockIdx == 0 &&
            (maxIdxInRst > this->elementsPerBlock || k > DEAL_TIMES_EACH_CORE_REDUCE)) {
            maxValueIndex += this->tailEle;
        }
        // Implementation note.
        this->rMinALLRstsTenor.SetValue(0, this->rMinDstTenor.GetValue(0));
        this->rMinALLRstsTenor.SetValue(1, *reinterpret_cast<float *>(&maxValueIndex));

        // Implementation note.
        event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID2);
        WaitFlag<HardEvent::S_V>(eventID2);
    }
}

template <typename T>
__aicore__ inline void Isamin<T>::getCoreTmpReduResult()
{
    LocalTensor<T> outLocalTensor = outDataQueue.AllocTensor<T>();

    // Implementation note.
    Copy(outLocalTensor, rMinALLRstsTenor, 8, 1, {1, 1, 8, 8});

    outDataQueue.EnQue(outLocalTensor);
}

template <typename T>
__aicore__ inline void Isamin<T>::CopyTmpRstToWkGM()
{
    LocalTensor<T> outLocalTensor = outDataQueue.DeQue<T>();
    uint16_t blockCout = (uint16_t)1;
    uint16_t blockLen = GM_RESULT_LEN * BYTE_LEN_4;
    DataCopyParams copyParams = {blockCout, blockLen, 0, 0};
    DataCopyPad(tmpRstWkGM, outLocalTensor, copyParams);

    outDataQueue.FreeTensor(outLocalTensor);
}

template <typename T>
__aicore__ inline void Isamin<T>::reduceMinCoresResult()
{
    copyInCoresTmpRst();
    ReduceMinCoresTmpRst();
    copyOutRst();
}

template <typename T>
__aicore__ inline void Isamin<T>::copyInCoresTmpRst()
{
    LocalTensor<T> inLocalTensor = coresRstInDataQueue.AllocTensor<T>();

    Duplicate<T>(inLocalTensor, FLT_MAX, this->coresRstReduceMinLen); // Implementation note.

    event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventID1);
    WaitFlag<HardEvent::V_MTE2>(eventID1);

    uint16_t blockLen = this->rstLenAllCoreBytes;
    DataCopyParams copyParams{1, blockLen, 0, 0};
    DataCopyPadParams padParams{true, 0, 0, 0}; // Implementation note.
    DataCopyPad(inLocalTensor, tmpRstWkGM, copyParams, padParams);

    coresRstInDataQueue.EnQue(inLocalTensor);
}

template <typename T>
__aicore__ inline void Isamin<T>::ReduceMinCoresTmpRst()
{
    LocalTensor<T> srcLocal = coresRstInDataQueue.DeQue<T>();
    LocalTensor<int32_t> outLocalTensor = coresRstOutDataQueue.AllocTensor<int32_t>();

    uint64_t mask[2] = {REDUCE_MAX_CORES_RST_MASK, 0};

    // Implementation note.
    ReduceMin(this->coresRstReduceTmp, srcLocal, this->coresRstReduceTmp, mask, 2, 8, true);

    event_t eventID3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID3);
    WaitFlag<HardEvent::V_S>(eventID3);

    float tmpK = this->coresRstReduceTmp.GetValue(1); // Implementation note.
    int32_t maxIdxInRst = *reinterpret_cast<uint32_t *>(&tmpK);
    float tmpK2 = srcLocal.GetValue(maxIdxInRst + 1); // Implementation note.
    int32_t maxValueIndex = *reinterpret_cast<uint32_t *>(&tmpK2) + 1; // Implementation note.

    outLocalTensor.SetValue(0, maxValueIndex);

    event_t eventIDSToMTE4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE4);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE4);

    coresRstOutDataQueue.EnQue(outLocalTensor);
    coresRstInDataQueue.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void Isamin<T>::copyOutRst()
{
    LocalTensor<int32_t> outLocalTensor = coresRstOutDataQueue.DeQue<int32_t>();

    uint16_t blockCout = (uint16_t)1;
    uint16_t blockLen = BYTE_LEN_4;
    DataCopyParams copyParams = {blockCout, blockLen, 0, 0};
    DataCopyPad(outTensorsGM, outLocalTensor, copyParams);
    coresRstOutDataQueue.FreeTensor(outLocalTensor);
}

template <typename T>
__aicore__ inline void Isamin<T>::getCoreReduResult()
{
    LocalTensor<int32_t> outLocalTensor = coresRstOutDataQueue.AllocTensor<int32_t>();

    float tmpK = rMinALLRstsTenor.GetValue(1);
    int32_t maxIdxInRst = *reinterpret_cast<uint32_t *>(&tmpK);  //
    rMinRstsTenor.SetValue(0, maxIdxInRst + 1); // Implementation note.

    event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID2);
    WaitFlag<HardEvent::S_V>(eventID2);

    Copy(outLocalTensor, rMinRstsTenor, 1, 1, {1, 1, 8, 8});

    coresRstOutDataQueue.EnQue(outLocalTensor);
}

template <typename T>
__aicore__ inline void Isamin<T>::CopyTmpRstOut()
{
    LocalTensor<int32_t> outLocalTensor = coresRstOutDataQueue.DeQue<int32_t>();

    uint16_t blockCout = (uint16_t)1;
    uint16_t blockLen = (uint16_t)1 * BYTE_LEN_4;
    DataCopyParams copyParams = {blockCout, blockLen, 0, 0};
    // Implementation note.
    DataCopyPad(outTensorsGM, outLocalTensor, copyParams);

    coresRstOutDataQueue.FreeTensor(outLocalTensor);
}
}  // namespace ISAMIN

#endif  // ISAMIN_H