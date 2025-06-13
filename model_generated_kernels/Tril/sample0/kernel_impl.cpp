#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲机制

class KernelTril {
public:
    __aicore__ inline KernelTril() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        // 设置全局内存指针
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        
        // 计算每个核处理的数据量
        uint32_t blockCount = GetBlockIdx() < 24 ? 2 : 1; // 前24个核处理2行，后24个核处理1行
        uint32_t rowOffset = GetBlockIdx() < 24 ? GetBlockIdx() * 2 : 24 * 2 + (GetBlockIdx() - 24) * 1;
        
        // 设置每个核处理的起始地址和长度
        uint32_t rowLength = 64; // 每行64个元素
        uint32_t tileLength = rowLength; // 每次处理整行
        
        // 初始化流水线缓冲区
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(float));
        
        this->rowOffset = rowOffset;
        this->tileLength = tileLength;
        this->rowLength = rowLength;
    }

    __aicore__ inline void Process() {
        // 双缓冲流水线处理
        for (int32_t i = 0; i < 2; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        // 从全局内存拷贝数据到本地缓冲区
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[this->rowOffset * this->rowLength + progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        // 获取本地缓冲区
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // 获取当前处理的行号
        uint32_t currentRow = this->rowOffset + progress;
        
        // 下三角计算：对当前行进行处理
        // 对于第currentRow行，列索引j >= currentRow的元素置零
        for (uint32_t j = 0; j < this->rowLength; ++j) {
            if (j >= currentRow) {
                yLocal.SetValue(j, (float)0.0);
            } else {
                yLocal.SetValue(j, xLocal.GetValue(j));
            }
        }
        
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        // 将结果写回全局内存
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[this->rowOffset * this->rowLength + progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t rowOffset;  // 当前核处理的起始行号
    uint32_t tileLength; // 每次处理的数据长度
    uint32_t rowLength;  // 每行的元素个数
};

extern "C" __global__ __aicore__ void tril(GM_ADDR x, GM_ADDR y, GM_ADDR workspace) {
    // 输入shape为[48,64]，float32类型
    // 使用24个AIC核和48个AIV核并行处理
    // 每个核处理2行数据（前24个核）或1行数据（后24个核）
    
    // 初始化tiling参数
    uint32_t totalLength = 48 * 64; // 总元素个数
    
    KernelTril op;
    op.Init(x, y, totalLength);
    op.Process();
}