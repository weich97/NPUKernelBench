#include "kernel_operator.h"
using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2;

class KernelLerp {
public:
    __aicore__ inline KernelLerp() {}
    
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, 
                              uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        startGm.SetGlobalBuffer((__gm__ float16_t*)start + this->blockLength * GetBlockIdx(), this->blockLength);
        endGm.SetGlobalBuffer((__gm__ float16_t*)end + this->blockLength * GetBlockIdx(), this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ float16_t*)weight + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float16_t*)y + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueStart, BUFFER_NUM, tileLength * sizeof(float16_t));
        pipe.InitBuffer(inQueueEnd, BUFFER_NUM, tileLength * sizeof(float16_t));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, tileLength * sizeof(float16_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(float16_t));
        pipe.InitBuffer(tempQueue, BUFFER_NUM, tileLength * sizeof(float16_t));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<float16_t> startLocal = inQueueStart.AllocTensor<float16_t>();
        LocalTensor<float16_t> endLocal = inQueueEnd.AllocTensor<float16_t>();
        LocalTensor<float16_t> weightLocal = inQueueWeight.AllocTensor<float16_t>();
        
        DataCopy(startLocal, startGm[progress * tileLength], tileLength);
        DataCopy(endLocal, endGm[progress * tileLength], tileLength);
        DataCopy(weightLocal, weightGm[progress * tileLength], tileLength);

        inQueueStart.EnQue(startLocal);
        inQueueEnd.EnQue(endLocal);
        inQueueWeight.EnQue(weightLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<float16_t> startLocal = inQueueStart.DeQue<float16_t>();
        LocalTensor<float16_t> endLocal = inQueueEnd.DeQue<float16_t>();
        LocalTensor<float16_t> weightLocal = inQueueWeight.DeQue<float16_t>();
        LocalTensor<float16_t> yLocal = outQueueY.AllocTensor<float16_t>();
        LocalTensor<float16_t> temp = tempQueue.AllocTensor<float16_t>();

        // 计算 end - start
        Sub(temp, endLocal, startLocal, tileLength);
        // 计算 (end - start) * weight
        Mul(temp, temp, weightLocal, tileLength);
        // 计算 start + (end - start)*weight
        Add(yLocal, startLocal, temp, tileLength);

        outQueueY.EnQue(yLocal);
        inQueueStart.FreeTensor(startLocal);
        inQueueEnd.FreeTensor(endLocal);
        inQueueWeight.FreeTensor(weightLocal);
        tempQueue.FreeTensor(temp);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<float16_t> yLocal = outQueueY.DeQue<float16_t>();
        DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueStart, inQueueEnd, inQueueWeight;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECIN, BUFFER_NUM> tempQueue;
    GlobalTensor<float16_t> startGm, endGm, weightGm, yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace) {
    TilingDataDef tiling_data = {
        .totalLength = 48 * 128 * 512,
        .tileNum = 4
    };

    KernelLerp op;
    op.Init(start, end, weight, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}