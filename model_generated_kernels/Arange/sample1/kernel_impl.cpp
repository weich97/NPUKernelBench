#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t SIZE_OF_INT32 = 4;
constexpr uint32_t BLOCK_SIZE = 32 * BUFFER_NUM / SIZE_OF_INT32;

__aicore__ inline int32_t CalculateLength(int32_t start, int32_t end, int32_t step, int32_t coreIdx) {
    int32_t totalLength = (end - start) / step;
    int32_t totalBlocks = (totalLength + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int32_t formerNum = totalBlocks % GetBlockNum();
    int32_t tailNum = GetBlockNum() - formerNum;
    int32_t formerLength = (totalLength / GetBlockNum() + 1) * BLOCK_SIZE;
    int32_t tailLength = (totalLength / GetBlockNum()) * BLOCK_SIZE;

    if (coreIdx < formerNum) {
        return formerLength;
    } else {
        return tailLength;
    }
}

__aicore__ inline void GenerateData(LocalTensor<int32_t>& dst, int32_t start, int32_t step, int32_t length) {
    int32_t value = start;
    for (int32_t i = 0; i < length; ++i) {
        dst.SetValue(i, value);
        value += step;
    }
}

extern "C" __global__ __aicore__ void arange(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out, GM_ADDR workspace)
{
    int32_t startVal = *((__gm__ int32_t*)start);
    int32_t endVal = *((__gm__ int32_t*)end);
    int32_t stepVal = *((__gm__ int32_t*)step);
    int32_t coreIdx = GetBlockIdx();
    int32_t length = CalculateLength(startVal, endVal, stepVal, coreIdx);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    pipe.InitBuffer(inQueue, BUFFER_NUM, length * sizeof(int32_t));

    LocalTensor<int32_t> outLocal = inQueue.AllocTensor<int32_t>();
    GenerateData(outLocal, startVal, stepVal, length);
    inQueue.EnQue(outLocal);

    GlobalTensor<int32_t> outGm;
    outGm.SetGlobalBuffer((__gm__ int32_t*)out + coreIdx * length, length);
    DataCopy(outGm, outLocal, length);

    inQueue.FreeTensor(outLocal);
}