#include "kernel_operator.h"

using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2;

class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * GetBlockIdx(), this->blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tempBuf, this->tileLength * sizeof(float));
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
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        LocalTensor<float> temp = tempBuf.Get<float>();

        const float sqrt_8_pi = 1.5957691216f;
        const float erf_coeff1 = 0.044715f;

        Mul(temp, xLocal, xLocal, this->tileLength);
        Mul(temp, temp, xLocal, this->tileLength);
        Muls(temp, temp, erf_coeff1, this->tileLength);
        Add(temp, xLocal, temp, this->tileLength);
        Muls(temp, temp, sqrt_8_pi, this->tileLength);
        Muls(temp, temp, -1.0f, this->tileLength);
        Exp(temp, temp, this->tileLength);
        Adds(temp, temp, 1.0f, this->tileLength);
        Div(zLocal, xLocal, temp, this->tileLength);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(yGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<TPosition::VECCALC> tempBuf;
    
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    TilingDataDef tiling_data = {
        .totalLength = 48 * 176,
        .tileNum = 1,
    };

    KernelGelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}