#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲队列
constexpr uint32_t SIZE_OF_HALF = 2; // float16类型大小
constexpr uint32_t BLOCK_SIZE = 32 * BUFFER_NUM / SIZE_OF_HALF; // 32字节对齐的分块大小

typedef struct {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t formerNum;
    uint32_t tailNum;
    uint32_t formerLength;
    uint32_t tailLength;
} TilingDataDef;

class KernelRsqrt {
public:
    __aicore__ inline KernelRsqrt() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TilingDataDef tiling_data) {
        // 根据核索引分配数据分片
        if (AscendC::GetBlockIdx() < tiling_data.formerNum) {
            this->blockLength = tiling_data.formerLength;
            uint32_t offset = tiling_data.formerLength * AscendC::GetBlockIdx();
            xGm.SetGlobalBuffer((__gm__ float16_t*)x + offset, this->blockLength);
            yGm.SetGlobalBuffer((__gm__ float16_t*)y + offset, this->blockLength);
        } else {
            this->blockLength = tiling_data.tailLength;
            uint32_t offset = tiling_data.formerLength * tiling_data.formerNum + 
                            tiling_data.tailLength * (AscendC::GetBlockIdx() - tiling_data.formerNum);
            xGm.SetGlobalBuffer((__gm__ float16_t*)x + offset, this->blockLength);
            yGm.SetGlobalBuffer((__gm__ float16_t*)y + offset, this->blockLength);
        }
        this->tileLength = BLOCK_SIZE; // 每个分块512个元素
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float16_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float16_t));
    }

    __aicore__ inline void Process() {
        // 分块处理数据
        for (uint32_t i = 0; i < this->blockLength; i += this->tileLength) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress) {
        // 从Global Memory搬运数据到UB
        LocalTensor<float16_t> xLocal = inQueueX.AllocTensor<float16_t>();
        DataCopy(xLocal, xGm[progress], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t progress) {
        // 执行平方根倒数计算
        LocalTensor<float16_t> xLocal = inQueueX.DeQue<float16_t>();
        LocalTensor<float16_t> yLocal = outQueueY.AllocTensor<float16_t>();
        Rsqrt(yLocal, xLocal, this->tileLength);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        // 将计算结果写回Global Memory
        LocalTensor<float16_t> yLocal = outQueueY.DeQue<float16_t>();
        DataCopy(yGm[progress], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float16_t> xGm;
    GlobalTensor<float16_t> yGm;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void rsqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace) {
    // Tiling参数计算
    TilingDataDef tiling_data;
    tiling_data.totalLength = 48 * 128 * 512; // 总数据量
    tiling_data.blockLength = BLOCK_SIZE; // 每个分块512元素
    tiling_data.formerNum = (tiling_data.totalLength / BLOCK_SIZE) % AscendC::GetBlockNum(); // 前formerNum个核多处理一个分块
    tiling_data.tailNum = AscendC::GetBlockNum() - tiling_data.formerNum; // 剩余核
    tiling_data.formerLength = (tiling_data.totalLength / BLOCK_SIZE / AscendC::GetBlockNum() + 1) * BLOCK_SIZE; // 前formerNum个核处理的长度
    tiling_data.tailLength = (tiling_data.totalLength / BLOCK_SIZE / AscendC::GetBlockNum()) * BLOCK_SIZE; // 剩余核处理的长度

    KernelRsqrt op;
    op.Init(x, y, tiling_data);
    op.Process();
}