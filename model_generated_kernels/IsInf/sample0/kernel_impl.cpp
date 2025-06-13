#include "kernel_operator.h"
using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2;

class KernelIsInf {
public:
    __aicore__ inline KernelIsInf() {}
    
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR outputs, uint32_t totalLength, uint32_t tileNum) {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        inputGm.SetGlobalBuffer((__gm__ float16_t*)inputs + this->blockLength * GetBlockIdx(), this->blockLength);
        outputGm.SetGlobalBuffer((__gm__ bool*)outputs + this->blockLength * GetBlockIdx(), this->blockLength);
        
        pipe.InitBuffer(inQueue, BUFFER_NUM, tileLength * sizeof(float16_t));
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileLength * sizeof(bool));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<float16_t> inputLocal = inQueue.AllocTensor<float16_t>();
        DataCopy(inputLocal, inputGm[progress * tileLength], tileLength);
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<float16_t> inputLocal = inQueue.DeQue<float16_t>();
        LocalTensor<bool> outputLocal = outQueue.AllocTensor<bool>();
        LocalTensor<uint16_t> bitsTensor = inputLocal.ReinterpretCast<uint16_t>();
        uint16_t infMask = 0x7C00;  // 0111110000000000

        for (int i = 0; i < tileLength; ++i) {
            uint16_t bits = bitsTensor.GetValue(i);
            outputLocal.SetValue(i, ((bits & infMask) == infMask) && ((bits & 0x03FF) == 0));
        }

        outQueue.EnQue<bool>(outputLocal);
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<bool> outputLocal = outQueue.DeQue<bool>();
        DataCopy(outputGm[progress * tileLength], outputLocal, tileLength);
        outQueue.FreeTensor(outputLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<float16_t> inputGm;
    GlobalTensor<bool> outputGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void is_inf(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) 
{
    TilingDataDef tiling_data = {
        .totalLength = 48 * 128 * 512,
        .tileNum = 8
    };

    KernelIsInf op;
    op.Init(inputs, outputs, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}