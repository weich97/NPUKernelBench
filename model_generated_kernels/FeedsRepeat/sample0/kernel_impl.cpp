#include "kernel_operator.h"

using namespace AscendC;

typedef struct {
    uint32_t output_feeds_size;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2;

class KernelFeedsRepeat {
public:
    __aicore__ inline KernelFeedsRepeat() {}
    __aicore__ inline void Init(GM_ADDR feeds, GM_ADDR feeds_repeat_times, GM_ADDR y, uint32_t output_feeds_size)
    {
        this->output_feeds_size = output_feeds_size;
        xGm.SetGlobalBuffer((__gm__ float16_t*)feeds, 48 * 128);
        repeatGm.SetGlobalBuffer((__gm__ int32_t*)feeds_repeat_times, 48);
        yGm.SetGlobalBuffer((__gm__ float16_t*)y, output_feeds_size * 128);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, 48 * 128 * sizeof(float16_t));
        pipe.InitBuffer(inQueueRepeat, BUFFER_NUM, 48 * sizeof(int32_t));
        pipe.InitBuffer(zeroQueue, BUFFER_NUM, 128 * sizeof(float16_t));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float16_t> xLocal = inQueueX.AllocTensor<float16_t>();
        AscendC::DataCopy(xLocal, xGm, 48 * 128);
        inQueueX.EnQue(xLocal);

        AscendC::LocalTensor<int32_t> repeatLocal = inQueueRepeat.AllocTensor<int32_t>();
        AscendC::DataCopy(repeatLocal, repeatGm, 48);
        inQueueRepeat.EnQue(repeatLocal);

        AscendC::LocalTensor<float16_t> zeroLocal = zeroQueue.AllocTensor<float16_t>();
        AscendC::Duplicate<float16_t>(zeroLocal, 0.0f, 128);
        zeroQueue.EnQue(zeroLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float16_t> xLocal = inQueueX.DeQue<float16_t>();
        AscendC::LocalTensor<int32_t> repeatLocal = inQueueRepeat.DeQue<int32_t>();
        
        sum_repeat = 0;
        for (int i = 0; i < 48; ++i) {
            sum_repeat += repeatLocal.GetValue(i);
        }

        uint32_t current_index = 0;
        for (int i = 0; i < 48; ++i) {
            int32_t repeats = repeatLocal.GetValue(i);
            for (int j = 0; j < repeats; ++j) {
                AscendC::DataCopy(yGm[current_index * 128], xLocal[i * 128], 128);
                ++current_index;
            }
        }

        inQueueX.FreeTensor(xLocal);
        inQueueRepeat.FreeTensor(repeatLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float16_t> zeroLocal = zeroQueue.DeQue<float16_t>();
        
        for (uint32_t i = sum_repeat; i < output_feeds_size; ++i) {
            AscendC::DataCopy(yGm[i * 128], zeroLocal, 128);
        }

        zeroQueue.FreeTensor(zeroLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueRepeat;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> zeroQueue;
    AscendC::GlobalTensor<float16_t> xGm;
    AscendC::GlobalTensor<int32_t> repeatGm;
    AscendC::GlobalTensor<float16_t> yGm;
    uint32_t output_feeds_size;
    uint32_t sum_repeat;
};

extern "C" __global__ __aicore__ void feeds_repeat(GM_ADDR feeds, GM_ADDR feeds_repeat_times, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TilingDataDef tiling_data = {
        .output_feeds_size = 1024,
    };
    KernelFeedsRepeat op;
    op.Init(feeds, feeds_repeat_times, y, tiling_data.output_feeds_size);
    op.Process();
}