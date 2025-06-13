#include "kernel_operator.h"

using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelSqrt {
public:
    __aicore__ inline KernelSqrt() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float16_t *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float16_t *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float16_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float16_t));
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
        AscendC::LocalTensor<float16_t> xLocal = inQueueX.AllocTensor<float16_t>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float16_t> xLocal = inQueueX.DeQue<float16_t>();
        AscendC::LocalTensor<float16_t> zLocal = outQueueZ.AllocTensor<float16_t>();
        AscendC::Sqrt(zLocal, xLocal, this->tileLength);
        outQueueZ.EnQue<float16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float16_t> zLocal = outQueueZ.DeQue<float16_t>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float16_t> xGm;
    AscendC::GlobalTensor<float16_t> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sqrt(GM_ADDR x, GM_ADDR z, GM_ADDR workspace)
{
    TilingDataDef tiling_data = {
        .totalLength = 48 * 128 * 512,
        .tileNum = 2,
    };
    KernelSqrt op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}