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
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        this->tileLength = blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float*)x + blockLength * AscendC::GetBlockIdx(), blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + blockLength * AscendC::GetBlockIdx(), blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, tileLength * sizeof(float));  // 临时缓冲区用于x^3计算
        pipe.InitBuffer(tmpBuffer2, tileLength * sizeof(float));  // 临时缓冲区用于中间结果
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

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        LocalTensor<float> tmp1 = tmpBuffer1.Get<float>();
        LocalTensor<float> tmp2 = tmpBuffer2.Get<float>();

        // GELU计算：y = x / (1 + exp(-sqrt(8/pi)*(x + 0.044715*x^3)))
        // 步骤1: 计算x^3 = x * x * x
        Mul(tmp1, xLocal, xLocal, tileLength);  // x^2
        Mul(tmp1, tmp1, xLocal, tileLength);    // x^3
        
        // 步骤2: 0.044715 * x^3
        Muls(tmp1, tmp1, 0.044715f, tileLength);
        
        // 步骤3: x + 0.044715*x^3 -> 存储在tmp1中
        Add(tmp1, xLocal, tmp1, tileLength);
        
        // 步骤4: 乘以sqrt(8/pi) ≈ 1.5957691216
        Muls(tmp1, tmp1, 1.5957691216f, tileLength);
        
        // 步骤5: 取负数
        Muls(tmp1, tmp1, -1.0f, tileLength);
        
        // 步骤6: 计算exp(-sqrt(8/pi)*(x + 0.044715*x^3))
        Exp(tmp1, tmp1, tileLength);
        
        // 步骤7: 加1 -> 1 + exp(...)
        Adds(tmp1, tmp1, 1.0f, tileLength);
        
        // 步骤8: 分子x除以分母
        Div(yLocal, xLocal, tmp1, tileLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
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
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;
    GlobalTensor<float> xGm, yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TilingDataDef tiling_data = {
        .totalLength = 8192,
        .tileNum = 8,
    };
    KernelGelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}