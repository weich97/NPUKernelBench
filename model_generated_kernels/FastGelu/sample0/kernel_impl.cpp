#include "kernel_operator.h"
using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2;

class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueTmp1, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueTmp2, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueTmp3, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
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

        LocalTensor<float> tmp_abs_x = inQueueTmp1.AllocTensor<float>();
        LocalTensor<float> tmp_denominator = inQueueTmp2.AllocTensor<float>();
        LocalTensor<float> tmp_exp_part = inQueueTmp3.AllocTensor<float>();

        Abs(tmp_abs_x, xLocal, tileLength);
        Muls(tmp_denominator, tmp_abs_x, -1.702f, tileLength);
        Exp(tmp_denominator, tmp_denominator, tileLength);
        Adds(tmp_denominator, tmp_denominator, 1.0f, tileLength);
        Div(zLocal, xLocal, tmp_denominator, tileLength);

        Sub(tmp_exp_part, xLocal, tmp_abs_x, tileLength);
        Muls(tmp_exp_part, tmp_exp_part, 0.851f, tileLength);
        Exp(tmp_exp_part, tmp_exp_part, tileLength);
        Mul(zLocal, zLocal, tmp_exp_part, tileLength);

        inQueueX.FreeTensor(xLocal);
        inQueueTmp1.FreeTensor(tmp_abs_x);
        inQueueTmp2.FreeTensor(tmp_denominator);
        inQueueTmp3.FreeTensor(tmp_exp_part);
        outQueueZ.EnQue(zLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(yGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueTmp1, inQueueTmp2, inQueueTmp3;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    TilingDataDef tiling_data = {
        .totalLength = 48 * 256,
        .tileNum = 1,
    };
    KernelFastGelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}