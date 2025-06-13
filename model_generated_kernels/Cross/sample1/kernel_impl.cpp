#include "kernel_operator.h"
using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2;

class KernelCross {
public:
    __aicore__ inline KernelCross() {}
    
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, uint32_t tileNum) {
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        this->tileLength = totalLength;

        x1Gm.SetGlobalBuffer((__gm__ float*)x1, this->blockLength);
        x2Gm.SetGlobalBuffer((__gm__ float*)x2, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, this->blockLength);
        
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<float> x1Local = inQueueX1.AllocTensor<float>();
        LocalTensor<float> x2Local = inQueueX2.AllocTensor<float>();
        DataCopy(x1Local, x1Gm, this->tileLength);
        DataCopy(x2Local, x2Gm, this->tileLength);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
        LocalTensor<float> x2Local = inQueueX2.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // 计算交叉乘积：y = x1 × x2
        // 对于每组3个元素，独立计算叉积
        for (int i = 0; i < this->tileLength; i += 3) {
            // 计算三个分量
            float cross0 = x1Local.GetValue(i+1) * x2Local.GetValue(i+2) - x1Local.GetValue(i+2) * x2Local.GetValue(i+1);
            float cross1 = x1Local.GetValue(i+2) * x2Local.GetValue(i+0) - x1Local.GetValue(i+0) * x2Local.GetValue(i+2);
            float cross2 = x1Local.GetValue(i+0) * x2Local.GetValue(i+1) - x1Local.GetValue(i+1) * x2Local.GetValue(i+0);
            
            yLocal.SetValue(i+0, cross0);
            yLocal.SetValue(i+1, cross1);
            yLocal.SetValue(i+2, cross2);
        }
        
        outQueueY.EnQue(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm, yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> x1Gm;
    GlobalTensor<float> x2Gm;
    GlobalTensor<float> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void cross(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace) {
    TilingDataDef tiling_data = {
        .totalLength = 48 * 3,
        .tileNum = 1,
    };
    
    KernelCross op;
    op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}