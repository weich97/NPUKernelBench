#include "kernel_operator.h"
using namespace AscendC;

namespace AscendC {
template <typename T>
class MulSigmoidMulAddCustom {
public:
    __aicore__ inline MulSigmoidMulAddCustom(){}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR mulScalar1, GM_ADDR mulScalar2, GM_ADDR addScalar3, GM_ADDR output, const MulSigmoidMulAddCustomTilingData &tiling)
    {
        this->mulScalar1 = *((__gm__ T*)mulScalar1);
        this->mulScalar2 = *((__gm__ T*)mulScalar2);
        this->addScalar3 = *((__gm__ T*)addScalar3);

        this->blockIdx = AscendC::GetBlockIdx();

        this->completeTileNum = tiling.completeTileNum;
        this->partTileNum = tiling.partTileNum;
        this->completeTileLen = tiling.completeTileLen;
        this->partTileLen = tiling.partTileLen;
        this->totalTileNum = tiling.totalTileNum;

        // Implementation note.
        if (blockIdx < tiling.frontBlockNum) {
            this->startTileIdx = blockIdx * tiling.tileNumInFrontBlock;
            this->tileNumInBlock = tiling.tileNumInFrontBlock;
        } else {
            this->startTileIdx = (tiling.frontBlockNum * tiling.tileNumInFrontBlock) + (blockIdx - tiling.frontBlockNum) * tiling.tileNumInLatterBlock;
            this->tileNumInBlock = tiling.tileNumInLatterBlock;
        }
        this->endTileIdx = this->startTileIdx + this->tileNumInBlock - 1;

        // Implementation note.
        this->startPosInBlock = this->startTileIdx * tiling.completeTileLen;
        if (this->endTileIdx < tiling.completeTileNum) {                
            this->blockLen = this->tileNumInBlock * tiling.completeTileLen;
        } else {
            this->blockLen = (this->tileNumInBlock - 1) * tiling.completeTileLen + tiling.partTileLen;
        }

        // Implementation note.
        inputGlobal.SetGlobalBuffer((__gm__ T*)input + this->startPosInBlock, this->blockLen);
        outputGlobal.SetGlobalBuffer((__gm__ T*)output + this->startPosInBlock * this->out_put_len, this->blockLen * this->out_put_len);

        // Implementation note.
        pipe.InitBuffer(inQueueInput, 2, this->completeTileLen * sizeof(T));
        pipe.InitBuffer(outQueueOutput, 2, this->completeTileLen * sizeof(T) * this->out_put_len);

        // Implementation note.
        pipe.InitBuffer(calcDataBuf, 2 * 2 * this->completeTileLen * sizeof(T));
    }

     __aicore__ inline void Process()
    {     
        for (int32_t tileIdx = this->startTileIdx; tileIdx <= this->endTileIdx; tileIdx++) {
            int32_t tileLen = (tileIdx < this->completeTileNum) ? this->completeTileLen : this->partTileLen;
            CopyIn(tileIdx, tileLen);
            Compute(tileIdx, tileLen);
            CopyOut(tileIdx, tileLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t tileIdx, int32_t tileLen)
    {
        // Implementation note.
        LocalTensor<T> inputLocal = inQueueInput.AllocTensor<T>();
        DataCopy(inputLocal, inputGlobal[tileIdx * tileLen], tileLen);
        inQueueInput.EnQue(inputLocal);
    }

    __aicore__ inline void CopyOut(int32_t tileIdx, int32_t tileLen)
    {
        LocalTensor<T> outputLocal = outQueueOutput.DeQue<T>();
        DataCopy(outputGlobal[tileIdx * tileLen], outputLocal, tileLen * this->out_put_len);
        outQueueOutput.FreeTensor(outputLocal);
    }

    __aicore__ inline void Compute(int32_t tileIdx, int32_t tileLen)
    {       
        // Implementation note.
        LocalTensor<T> inputLocal = inQueueInput.DeQue<T>();
        LocalTensor<T> outputLocal = outQueueOutput.AllocTensor<T>();

        // Implementation note.
        LocalTensor<T> calBuffer1 = calcDataBuf.Get<T>(tileLen);
        LocalTensor<T> calBuffer2 = calcDataBuf.GetWithOffset<T>(tileLen, tileLen);

        // Implementation note.
        AscendC::Muls(calBuffer1, inputLocal, this->mulScalar1, tileLen);

        // Implementation note.
        AscendC::Sigmoid(calBuffer2, calBuffer1, tileLen);

        // Implementation note.
        AscendC::Muls(calBuffer1, calBuffer2, this->mulScalar2, tileLen);

        // Implementation note.
        AscendC::Adds(outputLocal, calBuffer1, this->addScalar3, tileLen);
        inQueueInput.FreeTensor(inputLocal);
        outQueueOutput.EnQue<T>(outputLocal);
    }

private:
    TPipe pipe;

    // Implementation note.
    // Implementation note.
    // Implementation note.
    TQue<QuePosition::VECIN, 1> inQueueInput;
    TQue<QuePosition::VECOUT, 1> outQueueOutput;

    TBuf<QuePosition::VECCALC> calcDataBuf;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;

    uint32_t blockIdx;

    uint32_t completeTileNum;
    uint32_t partTileNum;
    uint32_t completeTileLen;
    uint32_t partTileLen;
    uint32_t totalTileNum;

    uint32_t startTileIdx;
    uint32_t tileNumInBlock;
    uint32_t endTileIdx;

    uint32_t startPosInBlock;
    uint32_t blockLen;

    T mulScalar1;
    T mulScalar2;
    T addScalar3;

    // Implementation note.
    uint32_t out_put_len = 1;
};
}


extern "C" __global__ __aicore__ 
void mul_sigmoid_mul_add_custom(GM_ADDR input, GM_ADDR mulScalar1, GM_ADDR mulScalar2, GM_ADDR addScalar3, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    MulSigmoidMulAddCustom<float16_t> op;
    op.Init(input, mulScalar1, mulScalar2, addScalar3, output, tiling_data);
    op.Process();
}
