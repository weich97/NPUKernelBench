#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t FLOAT16_SIZE = 2; // float16_t占2字节
constexpr uint32_t BLOCK_SIZE = 16;  // 32字节对齐要求：32/2=16个元素/块

typedef struct {
    uint32_t totalElements;  // 总元素数量
    uint32_t blockSize;      // 每个处理块元素数量
    uint32_t blockCount;     // 总块数
    uint32_t coreCount;      // 使用的核数量
    uint32_t elementsPerCore;// 每个核处理元素数量
    uint32_t blocksPerCore;  // 每个核处理块数量
} TilingDataDef;

class KernelRsqrt {
public:
    __aicore__ inline KernelRsqrt() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TilingDataDef tilingData) {
        this->tilingData = tilingData;
        xGm.SetGlobalBuffer((__gm__ float16_t*)x);
        yGm.SetGlobalBuffer((__gm__ float16_t*)y);
        
        // 初始化管道缓冲区：每个队列分配2个缓冲区，每个缓冲区容纳一个块数据
        pipe.InitBuffer(inQueueX, 2, tilingData.blockSize * FLOAT16_SIZE);
        pipe.InitBuffer(outQueueY, 2, tilingData.blockSize * FLOAT16_SIZE);
        
        // 计算当前核的起始偏移量
        uint32_t coreId = GetBlockIdx();
        startOffset = coreId * tilingData.elementsPerCore;
    }

    __aicore__ inline void Process() {
        // 双缓冲流水线：处理2个块的并行搬运
        for (uint32_t i = 0; i < tilingData.blocksPerCore; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress) {
        // 从全局内存拷贝数据到UB
        LocalTensor<float16_t> xLocal = inQueueX.AllocTensor<float16_t>();
        DataCopy(xLocal, 
                xGm[startOffset + progress * tilingData.blockSize], 
                tilingData.blockSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t progress) {
        // 从输入队列获取数据
        LocalTensor<float16_t> xLocal = inQueueX.DeQue<float16_t>();
        
        // 执行平方根倒数计算
        LocalTensor<float16_t> yLocal = outQueueY.AllocTensor<float16_t>();
        Rsqrt(yLocal, xLocal, tilingData.blockSize);
        
        // 将结果放入输出队列
        outQueueY.EnQue<float16_t>(yLocal);
        
        // 释放输入缓冲区
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        // 从输出队列获取结果
        LocalTensor<float16_t> yLocal = outQueueY.DeQue<float16_t>();
        
        // 将结果写回全局内存
        DataCopy(yGm[startOffset + progress * tilingData.blockSize], 
                yLocal, 
                tilingData.blockSize);
        
        // 释放输出缓冲区
        outQueueY.FreeTensor(yLocal);
    }

private:
    TilingDataDef tilingData;
    GlobalTensor<float16_t> xGm, yGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX;
    TQue<QuePosition::VECOUT, 2> outQueueY;
    uint32_t startOffset;
};

extern "C" __global__ __aicore__ void rsqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    // 计算分块参数
    TilingDataDef tilingData;
    tilingData.totalElements = 48 * 128 * 512;  // 总元素数
    tilingData.blockSize = BLOCK_SIZE;          // 每个处理块元素数
    tilingData.blockCount = tilingData.totalElements / tilingData.blockSize;
    tilingData.coreCount = GetBlockNum();       // 获取实际使用的核数量
    tilingData.elementsPerCore = tilingData.totalElements / tilingData.coreCount;
    tilingData.blocksPerCore = tilingData.elementsPerCore / tilingData.blockSize;

    // 实例化并执行计算
    KernelRsqrt op;
    op.Init(x, y, tilingData);
    op.Process();
}