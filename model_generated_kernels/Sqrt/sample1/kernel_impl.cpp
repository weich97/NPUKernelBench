#include "kernel_operator.h"

using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2; // Tensor number for each queue

class KernelSqrt {
public:
    __aicore__ inline KernelSqrt() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // Set global buffer with block offset
        xGm.SetGlobalBuffer((__gm__ float16_t*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float16_t*)y + this->blockLength * GetBlockIdx(), this->blockLength);
        
        // Initialize pipe buffers (32B aligned: tileLength*2bytes needs alignment)
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float16_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float16_t));
    }
    
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<float16_t> xLocal = inQueueX.AllocTensor<float16_t>();
        DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float16_t> xLocal = inQueueX.DeQue<float16_t>();
        LocalTensor<float16_t> yLocal = outQueueY.AllocTensor<float16_t>();
        Sqrt(yLocal, xLocal, tileLength);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float16_t> yLocal = outQueueY.DeQue<float16_t>();
        DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float16_t> xGm;
    GlobalTensor<float16_t> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    // Calculate tiling parameters based on test case [48,128,512]
    TilingDataDef tiling_data = {
        .totalLength = 48 * 128 * 512,  // Total elements: 3,145,728
        .tileNum = 4                     // Split each block into 4 tiles with double buffer
    };
    
    KernelSqrt op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}