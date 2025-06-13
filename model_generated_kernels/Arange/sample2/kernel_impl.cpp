#include "kernel_operator.h"

using namespace AscendC;

typedef struct {
    uint32_t outputLength;
    uint32_t tileLength;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t SIZE_OF_INT32 = 4;
constexpr uint32_t BLOCK_SIZE = 32 * BUFFER_NUM / SIZE_OF_INT32;

class KernelArange {
public:
    __aicore__ inline KernelArange() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out, TilingDataDef tiling_data)
    {
        this->start = (*(__gm__ int32_t*)start);
        this->end = (*(__gm__ int32_t*)end);
        this->step = (*(__gm__ int32_t*)step);
        this->tileLength = tiling_data.tileLength;
        this->outputLength = tiling_data.outputLength;
        outGm.SetGlobalBuffer((__gm__ int32_t*)out);
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(int32_t));
    }
    __aicore__ inline void Process()
    {
        int32_t current = this->start;
        int32_t progress = 0;
        while (progress < this->outputLength) {
            int32_t copyLength = (progress + this->tileLength > this->outputLength) ? 
                               (this->outputLength - progress) : this->tileLength;
            
            LocalTensor<int32_t> outLocal = outQueue.AllocTensor<int32_t>();
            for (int32_t i = 0; i < copyLength; i++) {
                outLocal.SetValue(i, current);
                current += this->step;
            }
            DataCopy(outGm[progress], outLocal, copyLength * sizeof(int32_t));
            outQueue.FreeTensor(outLocal);
            progress += copyLength;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<int32_t> outGm;
    int32_t start;
    int32_t end;
    int32_t step;
    uint32_t tileLength;
    uint32_t outputLength;
};

extern "C" __global__ __aicore__ void arange(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out, GM_ADDR workspace)
{
    TilingDataDef tiling_data;
    tiling_data.outputLength = (4096 - 1) / 1; // (end - start)/step
    tiling_data.tileLength = 256; // 256 elements per block (256*4=1024 bytes aligned)
    
    KernelArange op;
    op.Init(start, end, step, out, tiling_data);
    op.Process();
}