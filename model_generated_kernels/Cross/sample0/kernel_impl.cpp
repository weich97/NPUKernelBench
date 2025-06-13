#include "kernel_operator.h"

using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    uint32_t elementsPerCore;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 1;

class KernelCross {
public:
    __aicore__ inline KernelCross() {}
    
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, uint32_t elementsPerCore) {
        this->elementsPerCore = elementsPerCore;
        x1Gm.SetGlobalBuffer((__gm__ float*)x1 + elementsPerCore * GetBlockIdx(), elementsPerCore);
        x2Gm.SetGlobalBuffer((__gm__ float*)x2 + elementsPerCore * GetBlockIdx(), elementsPerCore);
        yGm.SetGlobalBuffer((__gm__ float*)y + elementsPerCore * GetBlockIdx(), elementsPerCore);
        
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, elementsPerCore * sizeof(float));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, elementsPerCore * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, elementsPerCore * sizeof(float));
    }

    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> x1Local = inQueueX1.AllocTensor<float>();
        LocalTensor<float> x2Local = inQueueX2.AllocTensor<float>();
        
        DataCopy(x1Local, x1Gm, elementsPerCore);
        DataCopy(x2Local, x2Gm, elementsPerCore);
        
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
        LocalTensor<float> x2Local = inQueueX2.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        for (uint32_t i = 0; i < elementsPerCore; i += 3) {
            float a1 = x1Local.GetValue(i);
            float a2 = x1Local.GetValue(i+1);
            float a3 = x1Local.GetValue(i+2);
            
            float b1 = x2Local.GetValue(i);
            float b2 = x2Local.GetValue(i+1);
            float b3 = x2Local.GetValue(i+2);

            yLocal.SetValue(i, a2*b3 - a3*b2);
            yLocal.SetValue(i+1, a3*b1 - a1*b3);
            yLocal.SetValue(i+2, a1*b2 - a2*b1);
        }

        outQueueY.EnQue(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm, yLocal, elementsPerCore);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> x1Gm, x2Gm, yGm;
    uint32_t elementsPerCore;
};

extern "C" __global__ __aicore__ void cross(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) 
{
    TilingDataDef tiling_data = {
        .totalLength = 48 * 3,
        .elementsPerCore = 48 * 3
    };
    
    KernelCross op;
    op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.elementsPerCore);
    op.Process();
}