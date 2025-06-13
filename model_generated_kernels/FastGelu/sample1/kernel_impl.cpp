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

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpQueue1, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpQueue2, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
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
        DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
    }
    
    __aicore__ inline void Compute( int32_t progress)
    {
        // x -> input
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        // 1: 计算|x|
        LocalTensor<float> absX = tmpQueue1.AllocTensor<float>();
        Abs(absX, xLocal, tileLength);
        // 2: 计算-1.702|x|
        LocalTensor<float> mulsResult = tmpQueue2.AllocTensor<float>();
        Muls(mulsResult, absX, (float)-1.702, tileLength);
        // 3: 计算exp(-1.702|x|)
        Exp(mulsResult, mulsResult, tileLength);
        // 4: 1 + exp_result
        Adds(mulsResult, mulsResult, (float)1.0, tileLength);
        // 5: x/(1 + exp(-1.702|x|))
        LocalTensor<float> divResult = tmpQueue1.AllocTensor<float>();
        Div(divResult, xLocal, mulsResult, tileLength);
        // 6: 计算x-|x|
        Sub(mulsResult, xLocal, absX, tileLength);
        // 7: 0.851*(x-|x|)
        Muls(mulsResult, mulsResult, (float)0.851, tileLength);
        // 8: 计算exp(0.851(x-|x|))
        Exp(mulsResult, mulsResult, tileLength);
        // 9: 最终结果相乘
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        Mul(yLocal, divResult, mulsResult, tileLength);
        
        outQueueY.EnQue(yLocal);
        tmpQueue1.FreeTensor(absX);
        tmpQueue1.FreeTensor(divResult);
        tmpQueue2.FreeTensor(xLocal);
        tmpQueue2.FreeTensor(mulsResult);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> tmpQueue1;
    TQue<QuePosition::VECIN, BUFFER_NUM> tmpQueue2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace) {
    TilingDataDef tiling_data;
    tiling_data.totalLength = 48 * 256;
    tiling_data.tileNum = 1;
    KernelFastGelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}