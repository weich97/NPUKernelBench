#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelArange {
public:
    __aicore__ inline KernelArange() {}
    
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out) {
        // 读取标量输入
        startVal = (*(__gm__ int32_t*)start);
        endVal = (*(__gm__ int32_t*)end);
        stepVal = (*(__gm__ int32_t*)step);
        
        // 计算输出长度
        int32_t length = (endVal - startVal) / stepVal + 1;
        this->length = length;
        
        // 设置全局内存输出
        outGm.SetGlobalBuffer((__gm__ int32_t*)out, length);
        
        // 初始化UB缓冲区
        pipe.InitBuffer(outQueue, BUFFER_NUM, length * sizeof(int32_t));
    }

    __aicore__ inline void Process() {
        // 分配UB缓冲区
        LocalTensor<int32_t> outLocal = outQueue.AllocTensor<int32_t>();
        
        // 计算输出值
        for (int32_t i = 0; i < length; ++i) {
            outLocal.SetValue(i, startVal + i * stepVal);
        }
        
        // 数据搬运到全局内存
        DataCopy(outGm, outLocal, length);
        outQueue.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<int32_t> outGm;
    int32_t startVal, endVal, stepVal, length;
};

extern "C" __global__ __aicore__ void arange(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out, GM_ADDR workspace)
{
    KernelArange op;
    op.Init(start, end, step, out);
    op.Process();
}