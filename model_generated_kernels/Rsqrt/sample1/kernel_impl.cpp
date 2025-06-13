#include "kernel_operator.h"
using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    uint32_t formerNum;
    uint32_t tailNum;
    uint32_t formerLength;
    uint32_t tailLength;
} TilingDataDef;

constexpr uint32_t TILE_SIZE = 32; // 32 float16 elements per tile (64 bytes aligned)
constexpr uint32_t BUFFER_NUM = 2;

class KernelRsqrt {
public:
    __aicore__ inline KernelRsqrt() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TilingDataDef tiling_data) {
        uint32_t core_idx = GetBlockIdx();
        if (core_idx < tiling_data.formerNum) {
            this->blockLength = tiling_data.formerLength;
            uint32_t offset = tiling_data.formerLength * core_idx;
            xGm.SetGlobalBuffer((__gm__ float16_t*)x + offset, this->blockLength);
            yGm.SetGlobalBuffer((__gm__ float16_t*)y + offset, this->blockLength);
        } else {
            this->blockLength = tiling_data.tailLength;
            uint32_t offset = tiling_data.formerLength * tiling_data.formerNum 
                            + tiling_data.tailLength * (core_idx - tiling_data.formerNum);
            xGm.SetGlobalBuffer((__gm__ float16_t*)x + offset, this->blockLength);
            yGm.SetGlobalBuffer((__gm__ float16_t*)y + offset, this->blockLength);
        }
        
        this->tileCount = this->blockLength / TILE_SIZE;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_SIZE * sizeof(float16_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_SIZE * sizeof(float16_t));
    }
    
    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < tileCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress) {
        LocalTensor<float16_t> xLocal = inQueueX.AllocTensor<float16_t>();
        DataCopy(xLocal, xGm[progress * TILE_SIZE], TILE_SIZE);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t progress) {
        LocalTensor<float16_t> xLocal = inQueueX.DeQue<float16_t>();
        LocalTensor<float16_t> yLocal = outQueueY.AllocTensor<float16_t>();
        Rsqrt(yLocal, xLocal, TILE_SIZE);
        outQueueY.EnQue<float16_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        LocalTensor<float16_t> yLocal = outQueueY.DeQue<float16_t>();
        DataCopy(yGm[progress * TILE_SIZE], yLocal, TILE_SIZE);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float16_t> xGm;
    GlobalTensor<float16_t> yGm;
    uint32_t blockLength;
    uint32_t tileCount;
};

extern "C" __global__ __aicore__ void rsqrt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace) {
    TilingDataDef tiling_data;
    tiling_data.totalLength = 48 * 128 * 512;
    tiling_data.formerNum = (tiling_data.totalLength / TILE_SIZE) % GetBlockNum();
    tiling_data.tailNum = GetBlockNum() - tiling_data.formerNum;
    tiling_data.tailLength = (tiling_data.totalLength / TILE_SIZE / GetBlockNum()) * TILE_SIZE;
    tiling_data.formerLength = tiling_data.tailLength + TILE_SIZE;

    KernelRsqrt op;
    op.Init(x, y, tiling_data);
    op.Process();
}