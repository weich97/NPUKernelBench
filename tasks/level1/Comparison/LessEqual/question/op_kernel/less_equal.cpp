#include "kernel_operator.h"

using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        x1Gm.SetGlobalBuffer((__gm__ float16_t *)x1 + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        x2Gm.SetGlobalBuffer((__gm__ float16_t *)x2 + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ int8_t *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(float16_t));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(float16_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(int8_t));
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
        AscendC::LocalTensor<float16_t> x1Local = inQueueX1.AllocTensor<float16_t>();
        AscendC::LocalTensor<float16_t> x2Local = inQueueX2.AllocTensor<float16_t>();
        AscendC::DataCopy(x1Local, x1Gm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(x2Local, x2Gm[progress * this->tileLength], this->tileLength);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float16_t> x1Local = inQueueX1.DeQue<float16_t>();
        AscendC::LocalTensor<float16_t> x2Local = inQueueX2.DeQue<float16_t>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
        AscendC::LocalTensor<float16_t> oneTensor = tmpQueue.AllocTensor<float16_t>();
        AscendC::LocalTensor<float16_t> zeroTensor = tmpQueue.AllocTensor<float16_t>();
        AscendC::LocalTensor<float16_t> tmpTensor = tmpQueue.AllocTensor<float16_t>();

        AscendC::Compare(tmpTensor, x1Local, x2Local, AscendC::CMPMODE::LE, this->tileLength);
        AscendC::Duplicate(oneTensor, (float16_t)1.0, this->tileLength);
        AscendC::Duplicate(zeroTensor, (float16_t)0.0, this->tileLength);
        AscendC::Select(tmpTensor, oneTensor, zeroTensor, tmpTensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->tileLength);
        AscendC::Cast(yLocal, tmpTensor, AscendC::RoundMode::CAST_NONE, this->tileLength);

        tmpQueue.FreeTensor(oneTensor);
        tmpQueue.FreeTensor(zeroTensor);
        tmpQueue.FreeTensor(tmpTensor);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        outQueueY.EnQue(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.DeQue<int8_t>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::QuePosition::VECCALC, 1> tmpQueue;
    AscendC::GlobalTensor<float16_t> x1Gm;
    AscendC::GlobalTensor<float16_t> x2Gm;
    AscendC::GlobalTensor<int8_t> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TilingDataDef tiling_data = {
        .totalLength = 3145728,
        .tileNum = 1024,
    };
    KernelLessEqual op;
    op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}