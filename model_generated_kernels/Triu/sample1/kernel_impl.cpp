#include "kernel_operator.h"

using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    int32_t diagonal;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2;

class KernelTriu {
public:
    __aicore__ inline KernelTriu() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, int32_t diagonal, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->diagonal = diagonal;
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->tileNum * BUFFER_NUM; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        for (int i = 0; i < this->tileLength; ++i) {
            int32_t globalCol = progress * this->tileLength + i;
            if (globalCol >= static_cast<int32_t>(AscendC::GetBlockIdx() + this->diagonal)) {
                yLocal.SetValue(i, xLocal.GetValue(i));
            } else {
                yLocal.SetValue(i, 0.0f);
            }
        }
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t blockLength;
    int32_t diagonal;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void triu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    TilingDataDef tiling_data = {
        .totalLength = 48 * 64,
        .diagonal = 0,
        .tileNum = 4
    };
    
    KernelTriu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.diagonal, tiling_data.tileNum);
    op.Process();
}