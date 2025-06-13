#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelSwish
{
public:
    __aicore__ inline KernelSwish() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum, float scale)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum)
        {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
        }
        this->scale = -1.0f * scale;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(calcBuf1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(calcBuf2, this->tileDataNum * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++)
        {
            if (i == this->tileNum - 1)
            {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_X> yLocal = outQueueY.AllocTensor<DTYPE_X>();
        if constexpr (std::is_same_v<DTYPE_X, float>){
            Muls(yLocal, xLocal, (DTYPE_X)this->scale, this->processDataNum);
            Exp(yLocal, yLocal, this->processDataNum);
            Adds(yLocal, yLocal, (DTYPE_X)1.0f, this->processDataNum);
            Div(yLocal, xLocal, yLocal, this->processDataNum);
        }else if constexpr (!std::is_same_v<DTYPE_X, float>){
            LocalTensor<float> xLocalFp32 = calcBuf1.Get<float>();
            LocalTensor<float> yLocalFp32 = calcBuf2.Get<float>();
            
            Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Muls(yLocalFp32, xLocalFp32, this->scale, this->processDataNum);
            Exp(yLocalFp32, yLocalFp32, this->processDataNum);
            Adds(yLocalFp32, yLocalFp32, 1.0f, this->processDataNum);
            Div(yLocalFp32, xLocalFp32, yLocalFp32, this->processDataNum);
            Cast(yLocal, yLocalFp32, RoundMode::CAST_ROUND, this->processDataNum);
        }
        outQueueY.EnQue<DTYPE_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DTYPE_X> yLocal = outQueueY.DeQue<DTYPE_X>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcBuf1;
    TBuf<QuePosition::VECCALC> calcBuf2;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float scale;
};

extern "C" __global__ __aicore__ void swish(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
 
    KernelSwish op;
    op.Init(x, y,tiling_data.smallCoreDataNum, 
        tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
        tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
        tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
        tiling_data.tailBlockNum, tiling_data.scale);
    op.Process();
}