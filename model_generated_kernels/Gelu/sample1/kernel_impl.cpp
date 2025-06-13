#include "kernel_operator.h"

using namespace AscendC;

// Tiling数据结构
typedef struct {
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2; // 每个队列的Tensor数量
constexpr float approx_sqrt_8_pi = 1.5957691216f; // √(8/π)的近似值

class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // 计算每个核处理的数据长度（假设单核处理全部数据）
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        this->tileLength = blockLength / tileNum / BUFFER_NUM;

        // 设置GlobalTensor：整个输入由单个核处理
        xGm.SetGlobalBuffer((__gm__ float*)x, blockLength);
        zGm.SetGlobalBuffer((__gm__ float*)y, blockLength);
        
        // 初始化流水线队列，每个缓冲区大小=tileLength * sizeof(float)
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(float));
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
        // 从Global Memory搬运数据到UB
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入队列取出xLocal
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        
        // 分配输出张量（重用作为临时空间）
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        
        // GELU计算步骤
        // 1. x² = x * x
        Mul(zLocal, xLocal, xLocal, tileLength);
        
        // 2. 0.044715 * x²
        Muls(zLocal, zLocal, 0.044715f, tileLength);
        
        // 3. 0.044715 * x³ = 0.044715 * x² * x
        Mul(zLocal, zLocal, xLocal, tileLength);
        
        // 4. x + 0.044715 x³
        Add(zLocal, xLocal, zLocal, tileLength);
        
        // 5. 乘以√(8/pi)
        Muls(zLocal, zLocal, approx_sqrt_8_pi, tileLength);
        
        // 6. 取相反数
        Muls(zLocal, zLocal, -1.0f, tileLength);
        
        // 7. 计算指数
        Exp(zLocal, zLocal, tileLength);
        
        // 8. 加1.0
        Adds(zLocal, zLocal, 1.0f, tileLength);
        
        // 9. 计算倒数
        Reciprocal(zLocal, zLocal, tileLength);
        
        // 10. 乘以x得到最终结果
        Mul(zLocal, xLocal, zLocal, tileLength);
        
        // 将结果加入输出队列
        outQueueZ.EnQue<float>(zLocal);
        
        // 释放输入张量
        inQueueX.FreeTensor(xLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列取出结果并写回Global Memory
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm;
    GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

// 核函数入口
extern "C" __global__ __aicore__ void gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    // 初始化Tiling参数
    TilingDataDef tiling_data = {
        .totalLength = 8192,  // 8*1024
        .tileNum = 1,         // 单个分块
    };
    
    // 实例化并初始化Kernel
    KernelGelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    
    // 执行处理流程
    op.Process();
}