#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace Sscal {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

template <typename T>
class SscalAIV {
public:
    __aicore__ inline SscalAIV() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR workSpace, GM_ADDR tilingGm);
    __aicore__ inline void Process();
    __aicore__ inline void Compute(uint32_t dataCount);
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void SingleIteration(uint32_t currOffset, uint32_t dataCount);
    __aicore__ inline void SingleIterationAligned(uint32_t currOffset, uint32_t dataCount);

private:
    TPipe pipe;

    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    uint32_t n = 0;
    uint32_t useCoreNum = 0;
    uint32_t calNum = 0;
    uint32_t startOffset = 0;
    float alpha = 0.0;
    uint32_t maxDataCount = 0;
    uint32_t totalVecCoreNum = 40;
    int32_t vecIdx;
    int32_t blockNum;

    int elementsPerBlock = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
    int elementsPerRepeat = UB_BYTENUM_PER_REPEAT / BYTENUM_PER_FLOAT32;
};

template <typename T>
__aicore__ inline void SscalAIV<T>::Init(GM_ADDR x, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    this->blockNum = GetBlockNum();
    this->vecIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    inGM.SetGlobalBuffer((__gm__ T *)x, this->n);
    outGM.SetGlobalBuffer((__gm__ T *)x, this->n);

    this->maxDataCount = 45 * 1024 / BYTENUM_PER_FLOAT32;

    pipe.InitBuffer(inQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, maxDataCount * sizeof(T));
}

template <typename T>
__aicore__ inline void SscalAIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);

    this->n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf));
    this->useCoreNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + sizeof(uint32_t)));
    this->startOffset =
        (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 2 * sizeof(uint32_t) + sizeof(uint32_t) * this->vecIdx));
    this->calNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 2 * sizeof(uint32_t) +
                                         this->useCoreNum * sizeof(uint32_t) + sizeof(uint32_t) * this->vecIdx));
    this->alpha = (*(__gm__ float *)((__gm__ uint8_t *)tilingBuf + 2 * sizeof(uint32_t) +
                                     2 * this->useCoreNum * sizeof(uint32_t)));
}

template <typename T>
__aicore__ inline void SscalAIV<T>::SingleIteration(uint32_t currOffset, uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopy(inLocal, inGM[currOffset], dataCount);
    inQueue.EnQue<T>(inLocal);

    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    Compute(dataCount);

    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopy(outGM[currOffset], outLocal, dataCount);
    outQueue.FreeTensor(outLocal);

    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void SscalAIV<T>::SingleIterationAligned(uint32_t currOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyPad(inLocal, inGM[currOffset], copyParams, padParams);
    inQueue.EnQue<T>(inLocal);

    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    uint32_t calCount = ((dataCount - 1) / elementsPerBlock + 1) * elementsPerBlock;
    Compute(calCount);

    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopyPad(outGM[currOffset], outLocal, copyParams);
    outQueue.FreeTensor(outLocal);

    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void SscalAIV<T>::Compute(uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    Muls(outLocal, inLocal, this->alpha, dataCount);
    inQueue.FreeTensor(inLocal);
    outQueue.EnQue<T>(outLocal);
}

template <typename T>
__aicore__ inline void SscalAIV<T>::Process()
{
    if (this->calNum <= 0) {
        return;
    }

    uint32_t repeatTimes = this->calNum / this->maxDataCount;
    uint32_t remainNum = this->calNum % this->maxDataCount;

    uint32_t currOffset = this->startOffset;

    for (uint32_t i = 0; i < repeatTimes; i++) {
        SingleIteration(currOffset, this->maxDataCount);
        currOffset += this->maxDataCount;
    }

    if (remainNum > 0) {
        SingleIterationAligned(currOffset, remainNum);
    }
}

}  // namespace Sscal

extern "C" __global__ __aicore__ void sscal(GM_ADDR x, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    if (TILING_KEY_IS(0)) {
        Sscal::SscalAIV<float> op;
        op.Init(x, workSpace, tilingGm);
        op.Process();
    }
}
