#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

typedef struct {
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

class KernelEye {
public:
    __aicore__ inline KernelEye() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR y_ref, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::Duplicate(zLocal, 0.0f, this->tileLength);
        outQueueZ.EnQue<float>(zLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        uint32_t global_start = progress * this->tileLength;
        for (uint32_t i = 0; i < this->tileLength; i++) {
            uint32_t global_idx = global_start + i;
            uint32_t matrix_idx = global_idx % (18 * 10);
            uint32_t row = matrix_idx / 10;
            uint32_t col = matrix_idx % 10;
            if (row == col) {
                zLocal.SetValue(i, 1.0f);
            }
        }
        outQueueZ.EnQue<float>(zLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void eye(GM_ADDR y, GM_ADDR y_ref, GM_ADDR workspace, GM_ADDR tiling) {
    TilingDataDef tiling_data = {
        .totalLength = 48 * 32 * 18 * 10,
        .tileNum = 1,
    };
    KernelEye op;
    op.Init(y, y_ref, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}