#include "kernel_operator.h"

using namespace AscendC;

template<typename T>
__aicore__ inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}
template<typename T>
__aicore__ inline T ISMAX(T num, T rnd)
{
    return ((num) > (rnd)) ? (num) : (rnd);
}

constexpr uint32_t DEFAULT_MIN_BLOCK_SIZE = 32; // Implementation note.

// Implementation note.
struct SwiGluSingleTileOffsetParam {
    uint64_t splitVecGmOffset1 = 0; // Implementation note.
    uint64_t splitVecGmOffset2 = 0; // Implementation note.
    uint64_t indepVecGmoffset = 0; // Implementation note.
};

struct SwiGluCopyParam {
    uint16_t blockCount = 0; // Implementation note.
    uint16_t blockLen = 0; // Implementation note.
    uint16_t stride = 0; // Implementation note.
};

struct SwiGluSinlgeTileCopyParam {
    SwiGluCopyParam splitVecCopyParam;
    SwiGluCopyParam indepVecCopyParam;
};

// tiling for SwiGlu Vector on one VectorCore
struct SwigluSingleTilingKernel {
    uint32_t is32BAligned = 1; // Is 32-byte aligned for split colLen?
    uint64_t totalBlockLen = 0; // Implementation note.
    uint64_t combColLen = 0; // Implementation note.
    uint64_t colLen = 0; // Implementation note.
    uint64_t rowLen = 0; // Implementation note.

    uint32_t baseRowLen = 0; // for one tile in one core, Unit:element
    uint32_t baseColLen = 0; // for one tile in one core, Unit:element
    uint32_t tailRowLen = 0; // number of tail row in one core, Unit:element
    uint32_t tailColLen = 0; // number of column in one core, Unit:element

    uint32_t tileLength = 0;  // baseRowLen * baseColLen

    uint64_t rowTileNum = 0; // Implementation note.
    uint64_t colTileNum = 0; // Implementation note.
    uint64_t totalTileNum = 0; // Implementation note.

    uint64_t baseRowTileNum = 0; // Implementation note.
    uint64_t baseColTileNum = 0; // Implementation note.

    // Implementation note.
    uint64_t baseRowBaseColCalLen = 0;
    uint64_t baseRowTailColCalLen = 0;
    uint64_t tailRowBaseColCalLen = 0;
    uint64_t tailRowTailColCalLen = 0;
    SwiGluSinlgeTileCopyParam baseRowBaseColCopyParam;
    SwiGluSinlgeTileCopyParam baseRowTailColCopyParam;
    SwiGluSinlgeTileCopyParam tailRowBaseColCopyParam;
    SwiGluSinlgeTileCopyParam tailRowTailColCopyParam;

    // Implementation note.
    uint64_t curCalLen;
    SwiGluSingleTileOffsetParam offsetParam;
    SwiGluSinlgeTileCopyParam *curTileCopyParam = nullptr;

  // calc tiling data
  __aicore__ void GetTilingAndOffset(GM_ADDR tiling_gm, uint32_t inputDTypeLen) {
    GET_TILING_DATA(tempTilingGm, tiling_gm);

    is32BAligned = tempTilingGm.is32BAligned;
    rowLen = tempTilingGm.rowLen;
    colLen = tempTilingGm.colLen;
    // Implementation note.
    combColLen = colLen * 2;
    totalBlockLen = rowLen * combColLen;

    uint32_t minInputDTypeLen = 2;
    uint32_t isZero = 1;

    baseRowLen = tempTilingGm.baseRowLen;
    baseColLen = tempTilingGm.baseColLen;
    // Implementation note.
    tileLength = (is32BAligned == 1) ? (baseRowLen * baseColLen) :
        baseRowLen * AlignUp<uint32_t>(baseColLen,
        ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));

    // Implementation note.
    baseRowTileNum = rowLen / baseRowLen;
    baseColTileNum = colLen / baseColLen;
    tailRowLen = rowLen % baseRowLen;
    tailColLen = colLen % baseColLen;
    rowTileNum = (tailRowLen > 0) ? (baseRowTileNum + 1) : baseRowTileNum;
    colTileNum = (tailColLen > 0) ? (baseColTileNum + 1) : baseColTileNum;
    totalTileNum = rowTileNum * colTileNum;

    // Implementation note.
    CaclTileCopyParams(inputDTypeLen);
  }

  __aicore__ inline void CaclOneTileCopyParam(uint64_t calRowLen, uint64_t calColLen, uint32_t inputDTypeLen, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
  {
    // Implementation note.
    uint16_t blockUnit = (is32BAligned == 1) ? DEFAULT_MIN_BLOCK_SIZE : 1;
    SwiGluCopyParam.splitVecCopyParam.blockCount = calRowLen;
    SwiGluCopyParam.splitVecCopyParam.blockLen = calColLen * inputDTypeLen / blockUnit;
    SwiGluCopyParam.splitVecCopyParam.stride =
        calRowLen == 1 ? 0 : ((combColLen - calColLen) * inputDTypeLen / blockUnit);

    SwiGluCopyParam.indepVecCopyParam.blockCount = calRowLen;
    SwiGluCopyParam.indepVecCopyParam.blockLen = calColLen * inputDTypeLen / blockUnit;
    SwiGluCopyParam.indepVecCopyParam.stride =
        calRowLen == 1 ? 0 : ((colLen - calColLen) * inputDTypeLen / blockUnit);
  }

  // Implementation note.
  __aicore__ inline void CaclTileCopyParams(uint32_t inputDTypeLen) {
    // Implementation note.
    // zone1:baseRow-baseCol  zone2:baseRow-tailCol  zone3:tailRow-baseCol  zone4:tailRow-tailCol
    // base row , base col
    uint32_t minInputDTypeLen = 2;
    uint32_t isZero = 1;
    baseRowBaseColCalLen = (is32BAligned == 1) ? (baseRowLen * baseColLen) :
      (baseRowLen * AlignUp<uint32_t>(baseColLen,
      ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen)))));
    CaclOneTileCopyParam(baseRowLen, baseColLen, inputDTypeLen, baseRowBaseColCopyParam);

    // base row , tail col
    baseRowTailColCalLen = (is32BAligned == 1) ? (baseRowLen * tailColLen) :
      baseRowLen * AlignUp<uint32_t>(tailColLen,
      ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));
    CaclOneTileCopyParam(baseRowLen, tailColLen, inputDTypeLen, baseRowTailColCopyParam);

    // tail row , base col
    tailRowBaseColCalLen = (is32BAligned == 1) ? (tailRowLen * baseColLen) :
      tailRowLen * AlignUp<uint32_t>(baseColLen,
      ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));
    CaclOneTileCopyParam(tailRowLen, baseColLen, inputDTypeLen, tailRowBaseColCopyParam);

    // tail row , tail col
    tailRowTailColCalLen = (is32BAligned == 1) ? (tailRowLen * tailColLen) :
      tailRowLen * AlignUp<uint32_t>(tailColLen,
      ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));
    CaclOneTileCopyParam(tailRowLen, tailColLen, inputDTypeLen, tailRowTailColCopyParam);
  }

  // Implementation note.
  __aicore__ inline void CaclOneTileOffsetParam(uint64_t gmRowOffset, uint64_t colIdx)
  {
    // Implementation note.
    // Implementation note.
    offsetParam.splitVecGmOffset1 = gmRowOffset * combColLen + colIdx * baseColLen;
    // Implementation note.
    offsetParam.splitVecGmOffset2 = offsetParam.splitVecGmOffset1 + colLen;
    // Implementation note.
    offsetParam.indepVecGmoffset = gmRowOffset * colLen + colIdx * baseColLen;
  }

  __aicore__ inline void CaclOneTileParam(uint64_t tileIdx)
  {
    uint64_t rowTileIdx = tileIdx / colTileNum;
    uint64_t colTileIdx = tileIdx % colTileNum;
    CaclOneTileOffsetParam(rowTileIdx * baseRowLen, colTileIdx);
    if (rowTileIdx < baseRowTileNum) {
      if (colTileIdx < baseColTileNum) {
        // base row, base col
        curCalLen = baseRowBaseColCalLen;
        curTileCopyParam = &baseRowBaseColCopyParam;
      } else {
        // base row, tail col
        curCalLen = baseRowTailColCalLen;
        curTileCopyParam = &baseRowTailColCopyParam;
      }
    } else {
      if (colTileIdx < baseColTileNum) {
        // tail row, base col
        curCalLen = tailRowBaseColCalLen;
        curTileCopyParam = &tailRowBaseColCopyParam;
      } else {
        // tail row, tail col
        curCalLen = tailRowTailColCalLen;
        curTileCopyParam = &tailRowTailColCopyParam;
      }
    }
  }
};
#define SWIGLU_SINGLE_PROCESS_TILE(offsetParam, SwiGluCopyParam, calLen) \
do {                                \
    CopyIn(offsetParam, SwiGluCopyParam); \
    this->Compute(calLen); \
    CopyOut(offsetParam, SwiGluCopyParam); \
} while (0)
#define SWIGLU_SINGLE_PROCESS(kernelTiling) \
do {                                       \
    uint64_t blockNum = GetBlockNum();       \
	  for(uint64_t tileIdx = get_block_idx(); tileIdx < (kernelTiling).totalTileNum; tileIdx += blockNum) { \
        (kernelTiling).CaclOneTileParam(tileIdx); \
        SWIGLU_SINGLE_PROCESS_TILE((kernelTiling).offsetParam, *((kernelTiling).curTileCopyParam), (kernelTiling).curCalLen); \
	} \
} while(0)
#define SWIGLU_SINGLE_PROCESS_TILE_NON32BALIGNED(offsetParam, SwiGluCopyParam, calLen) \
do {                                              \
    CopyIn_Non32BAligned(offsetParam, SwiGluCopyParam); \
    this->Compute(calLen); \
    CopyOut_Non32BAligned(offsetParam, SwiGluCopyParam); \
} while(0)
#define SWIGLU_SINGLE_PROCESS_NON32BALIGNED(kernelTiling) \
do {                                       \
    uint64_t blockNum = GetBlockNum();       \
	  for(uint64_t tileIdx = get_block_idx(); tileIdx < (kernelTiling).totalTileNum; tileIdx += blockNum) { \
        (kernelTiling).CaclOneTileParam(tileIdx); \
        SWIGLU_SINGLE_PROCESS_TILE_NON32BALIGNED((kernelTiling).offsetParam, *((kernelTiling).curTileCopyParam), (kernelTiling).curCalLen); \
	} \
} while(0)

template<typename aType, typename bType, typename lType, typename mType, typename nType, uint16_t bufferNum>
class SwiGluGradVector {
public:
    __aicore__ inline SwiGluGradVector() {}

protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);

    float beta = 1.0;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECIN, bufferNum> inQueueL;
    TQue<QuePosition::VECOUT, bufferNum> outQueueM;
    TQue<QuePosition::VECOUT, bufferNum> outQueueN;
    TBuf<TPosition::VECCALC> tmpQueue;
    TBuf<TPosition::VECCALC> sigQueue;
    LocalTensor<float> tempLocal;
    LocalTensor<float> sigLocal;
    GlobalTensor<aType> aGm;
    GlobalTensor<bType> bGm;
    GlobalTensor<lType> lGm;
    GlobalTensor<mType> mGm;
    GlobalTensor<nType> nGm;
};

template<typename aType, typename bType, typename lType, typename mType, typename nType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradVector<aType, bType, lType, mType, nType, bufferNum>::InitUbBuffer(uint64_t tileLength)
{
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(aType));
    pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(bType));
    pipe.InitBuffer(inQueueL, bufferNum, tileLength * sizeof(lType));
    pipe.InitBuffer(outQueueM, bufferNum, tileLength * sizeof(mType)); // The length must be an integer multiple of 32
    pipe.InitBuffer(outQueueN, bufferNum, tileLength * sizeof(nType)); // The length must be an integer multiple of 32

    pipe.InitBuffer(tmpQueue, tileLength * sizeof(float));
    pipe.InitBuffer(sigQueue, tileLength * sizeof(float));
    tempLocal = tmpQueue.Get<float>();
    sigLocal = sigQueue.Get<float>();
}

template<typename aType, typename bType, typename lType, typename mType, typename nType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradVector<aType, bType, lType, mType, nType, bufferNum>::Compute(uint64_t tileLength)
{
    // tbuf::templocaltensor
    // deque input tensors from VECIN queue
    //calc sigLocal
    LocalTensor<aType> aLocal = inQueueA.template DeQue<aType>(); //input a
    Muls(sigLocal, aLocal, beta, tileLength);
    pipe_barrier(PIPE_V);
    Exp(sigLocal, sigLocal, tileLength);
    pipe_barrier(PIPE_V);
    Adds(sigLocal, sigLocal, (mType)(1.0), tileLength);
    pipe_barrier(PIPE_V);
    Duplicate<float>(tempLocal, (float)(1.0), tileLength);
    Div(sigLocal, tempLocal, sigLocal, tileLength);
    pipe_barrier(PIPE_V);

    //----------------N
    LocalTensor<nType> nLocal = outQueueN.template AllocTensor<nType>(); // lb
    Mul(nLocal, sigLocal, aLocal, tileLength);
    LocalTensor<lType> lLocal = inQueueL.template DeQue<lType>(); // input l
    Mul(nLocal, nLocal, lLocal, tileLength);
    pipe_barrier(PIPE_V);
    outQueueN.template EnQue<nType>(nLocal);

    //----------------M
    Muls(tempLocal, sigLocal, (mType)(-1.0), tileLength);
    pipe_barrier(PIPE_V);
    Adds(tempLocal, tempLocal, (mType)(1.0), tileLength);
    pipe_barrier(PIPE_V);
    LocalTensor<mType> mLocal = outQueueM.template AllocTensor<mType>(); // la
    Mul(mLocal, sigLocal, tempLocal, tileLength);
    Mul(mLocal, mLocal, aLocal, tileLength);
    inQueueA.template FreeTensor(aLocal);

    Muls(mLocal, mLocal, -beta, tileLength);
    Add(mLocal, mLocal, sigLocal, tileLength);
    LocalTensor<bType> bLocal = inQueueB.template DeQue<bType>(); //input b
    Mul(mLocal, mLocal, bLocal, tileLength);
    inQueueB.template FreeTensor(bLocal);

    Mul(mLocal, mLocal, lLocal, tileLength);
    pipe_barrier(PIPE_V);
    // enque the output tensor to VECOUT queue
    outQueueM.template EnQue<mType>(mLocal);
    // free input tensors for reuse
    inQueueL.template FreeTensor(lLocal);
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
class SwiGluGradBF16 {
  public:
    __aicore__ inline SwiGluGradBF16() {}

  protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);
    __aicore__ inline void ComputeSigLocal(uint64_t curTileLen);
    __aicore__ inline void ComputeGradN(uint64_t curTileLen);

    calcType beta = 1.0;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECIN, bufferNum> inQueueL;
    TQue<QuePosition::VECOUT, bufferNum> outQueueM;
    TQue<QuePosition::VECOUT, bufferNum> outQueueN;
    TBuf<TPosition::VECCALC> tmpQueue;
    TBuf<TPosition::VECCALC> sigQueue;
    LocalTensor<calcType> tempLocal;
    LocalTensor<calcType> sigLocal;
    LocalTensor<calcType> aLocal;
    LocalTensor<calcType> nLocal;
    LocalTensor<calcType> lLocal;

    TBuf<TPosition::VECCALC> aTempBuffer;
    TBuf<TPosition::VECCALC> lTempBuffer;
    TBuf<TPosition::VECCALC> outputTempBuffer;

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<inType> lGm;
    GlobalTensor<outType> mGm;
    GlobalTensor<outType> nGm;
};

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::InitUbBuffer(uint64_t tileLength)
{
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(inQueueL, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(outQueueM, bufferNum, tileLength * sizeof(outType)); // The length must be an integer multiple of 32
    pipe.InitBuffer(outQueueN, bufferNum, tileLength * sizeof(outType)); // The length must be an integer multiple of 32

    pipe.InitBuffer(tmpQueue, tileLength * sizeof(calcType));
    pipe.InitBuffer(sigQueue, tileLength * sizeof(calcType));

    pipe.InitBuffer(aTempBuffer, tileLength * sizeof(calcType));
    pipe.InitBuffer(lTempBuffer, tileLength * sizeof(calcType));
    pipe.InitBuffer(outputTempBuffer, tileLength * sizeof(calcType));
    tempLocal = tmpQueue.Get<calcType>();
    sigLocal = sigQueue.Get<calcType>();
    aLocal = aTempBuffer.Get<calcType>(); 
    nLocal = outputTempBuffer.Get<calcType>(); 
    lLocal = lTempBuffer.Get<calcType>();    
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::Compute(uint64_t tileLength)
{
    // tbuf::templocaltensor
    // deque input tensors from VECIN queue
    //calc sigLocal
    ComputeSigLocal(tileLength);
    //----------------N
    Mul(nLocal, sigLocal, aLocal, tileLength);
    ComputeGradN(tileLength);

    //----------------M
    Muls(tempLocal, sigLocal, (calcType)(-1.0), tileLength);
    pipe_barrier(PIPE_V);
    Adds(tempLocal, tempLocal, (calcType)(1.0), tileLength);
    pipe_barrier(PIPE_V);

    auto& mLocal = nLocal;                      
    Mul(mLocal, sigLocal, tempLocal, tileLength); 
    pipe_barrier(PIPE_V);
    Mul(mLocal, mLocal, aLocal, tileLength);  
    pipe_barrier(PIPE_V);

    Muls(mLocal, mLocal, -beta, tileLength);
    pipe_barrier(PIPE_V);
    Add(mLocal, mLocal, sigLocal, tileLength);

    LocalTensor<inType> bLocal_ = inQueueB.template DeQue<inType>();
    auto& bLocal = aLocal; 
    Cast(bLocal, bLocal_, RoundMode::CAST_NONE, tileLength);
    pipe_barrier(PIPE_V);

    Mul(mLocal, mLocal, lLocal, tileLength);
    pipe_barrier(PIPE_V);

    LocalTensor<outType> mLocal_ = outQueueM.template AllocTensor<outType>();

    Mul(mLocal, mLocal, bLocal, tileLength);
    inQueueB.template FreeTensor(bLocal_);

    Cast(mLocal_, mLocal, RoundMode::CAST_RINT, tileLength);
    pipe_barrier(PIPE_V);
    outQueueM.template EnQue<outType>(mLocal_);
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::ComputeSigLocal(uint64_t tileLength)
{
    // tbuf::templocaltensor
    // deque input tensors from VECIN queue
    //calc sigLocal
    LocalTensor<inType> aLocal_ = inQueueA.template DeQue<inType>(); //input a

    Cast(aLocal, aLocal_, RoundMode::CAST_NONE, tileLength);
    pipe_barrier(PIPE_V);
    inQueueA.template FreeTensor(aLocal_);

    Muls(sigLocal, aLocal, beta, tileLength);
    pipe_barrier(PIPE_V);
    Exp(sigLocal, sigLocal, tileLength);
    pipe_barrier(PIPE_V);
    Adds(sigLocal, sigLocal, (calcType)(1.0), tileLength);
    pipe_barrier(PIPE_V);
    Duplicate<calcType>(tempLocal, (calcType)(1.0), tileLength);
    pipe_barrier(PIPE_V);
    Div(sigLocal, tempLocal, sigLocal, tileLength);
    pipe_barrier(PIPE_V);
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::ComputeGradN(uint64_t tileLength)
{
    LocalTensor<inType> lLocal_ = inQueueL.template DeQue<inType>(); // input l
    Cast(lLocal, lLocal_, RoundMode::CAST_NONE, tileLength);
    pipe_barrier(PIPE_V);
    inQueueL.template FreeTensor(lLocal_);

    LocalTensor<outType> nLocal_ = outQueueN.template AllocTensor<outType>(); // lb

    Mul(nLocal, nLocal, lLocal, tileLength);
    pipe_barrier(PIPE_V);

    Cast(nLocal_, nLocal, RoundMode::CAST_RINT, tileLength); // Implementation note.
    pipe_barrier(PIPE_V);
    outQueueN.template EnQue<outType>(nLocal_);
}

using namespace AscendC;
template<typename ParentClass, typename inType, typename outType>
class SwiGluGradSingle : public ParentClass {
public:
    __aicore__ inline SwiGluGradSingle() = default;
    __aicore__ inline ~SwiGluGradSingle() = default;
    __aicore__ inline void Init(GM_ADDR grad_gm, GM_ADDR input_gm, GM_ADDR output_gm, GM_ADDR tiling_gm)
    {
        singleTiling.GetTilingAndOffset(tiling_gm, sizeof(inType));
        InitGmBuffer(grad_gm, input_gm, output_gm);
        this->InitUbBuffer(singleTiling.tileLength);
    }

    __aicore__ inline void Process()
    {
        if (singleTiling.is32BAligned == 1) {
            SWIGLU_SINGLE_PROCESS(singleTiling);
        } else {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            SWIGLU_SINGLE_PROCESS_NON32BALIGNED(singleTiling);
#endif
        }
    }
protected:
    __aicore__ inline void InitGmBuffer(GM_ADDR grad_gm, GM_ADDR input_gm, GM_ADDR output_gm)
    {
        // get start index for current core, core parallel
        this->beta = -1.0f;
        this->aGm.SetGlobalBuffer((__gm__ inType*)input_gm, singleTiling.totalBlockLen);
        this->lGm.SetGlobalBuffer((__gm__ inType*)grad_gm, singleTiling.totalBlockLen / 2);

        this->mGm.SetGlobalBuffer((__gm__ outType*)output_gm, singleTiling.totalBlockLen);
    }

    __aicore__ inline void CopyIn(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyinParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                            SwiGluCopyParam.splitVecCopyParam.blockLen,
                                            SwiGluCopyParam.splitVecCopyParam.stride,
                                            0};
        DataCopyParams indepCopyinParams = {SwiGluCopyParam.indepVecCopyParam.blockCount,
                                            SwiGluCopyParam.indepVecCopyParam.blockLen,
                                            SwiGluCopyParam.indepVecCopyParam.stride,
                                            0};

        // Copy A
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopy(aLocal, this->aGm[offsetParam.splitVecGmOffset1], splitCopyinParams);
        this->inQueueA.template EnQue(aLocal);
        // Copy L
        LocalTensor<inType> lLocal = this->inQueueL.template AllocTensor<inType>();
        DataCopy(lLocal, this->lGm[offsetParam.indepVecGmoffset], indepCopyinParams);
        this->inQueueL.template EnQue(lLocal);
        // Copy B
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopy(bLocal, this->aGm[offsetParam.splitVecGmOffset2], splitCopyinParams);
        this->inQueueB.template EnQue(bLocal);
    }

    __aicore__ inline void CopyOut(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyoutParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                             SwiGluCopyParam.splitVecCopyParam.blockLen,
                                             0,
                                             SwiGluCopyParam.splitVecCopyParam.stride};

        // deque output tensor from VECOUT queue
        LocalTensor<outType> mLocal = this->outQueueM.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(this->mGm[offsetParam.splitVecGmOffset1], mLocal, splitCopyoutParams);

        // free output tensor for reuse
        this->outQueueM.template FreeTensor(mLocal);

        // deque output tensor from VECOUT queue
        LocalTensor<outType> nLocal = this->outQueueN.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(this->mGm[offsetParam.splitVecGmOffset2], nLocal, splitCopyoutParams);

        // free output tensor for reuse
        this->outQueueN.template FreeTensor(nLocal);
    }

    __aicore__ inline void CopyIn_Non32BAligned(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyinParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                            SwiGluCopyParam.splitVecCopyParam.blockLen,
                                            SwiGluCopyParam.splitVecCopyParam.stride,
                                            0};
        DataCopyParams indepCopyinParams = {SwiGluCopyParam.indepVecCopyParam.blockCount,
                                            SwiGluCopyParam.indepVecCopyParam.blockLen,
                                            SwiGluCopyParam.indepVecCopyParam.stride,
                                            0};
        DataCopyPadParams copyPadParams = {false, 0, 0, 0};
        // Copy A
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopyPad(aLocal, this->aGm[offsetParam.splitVecGmOffset1], splitCopyinParams, copyPadParams);
        this->inQueueA.template EnQue(aLocal);
        // Copy L
        LocalTensor<inType> lLocal = this->inQueueL.template AllocTensor<inType>();
        DataCopyPad(lLocal, this->lGm[offsetParam.indepVecGmoffset], indepCopyinParams, copyPadParams);
        this->inQueueL.template EnQue(lLocal);
        // Copy B
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopyPad(bLocal, this->aGm[offsetParam.splitVecGmOffset2], splitCopyinParams, copyPadParams);
        this->inQueueB.template EnQue(bLocal);
    }

    __aicore__ inline void CopyOut_Non32BAligned(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyoutParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                             SwiGluCopyParam.splitVecCopyParam.blockLen,
                                             0,
                                             SwiGluCopyParam.splitVecCopyParam.stride};

        // deque output tensor from VECOUT queue
        LocalTensor<outType> mLocal = this->outQueueM.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopyPad(this->mGm[offsetParam.splitVecGmOffset1], mLocal, splitCopyoutParams);
        // free output tensor for reuse
        this->outQueueM.template FreeTensor(mLocal);

        // deque output tensor from VECOUT queue
        LocalTensor<outType> nLocal = this->outQueueN.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopyPad(this->mGm[offsetParam.splitVecGmOffset2], nLocal, splitCopyoutParams);
        // free output tensor for reuse
        this->outQueueN.template FreeTensor(nLocal);
    }

private:
    SwigluSingleTilingKernel singleTiling;
};

using namespace AscendC;
extern "C" __global__ __aicore__ void swi_glu_grad(GM_ADDR gradout_gm, GM_ADDR input_gm, GM_ADDR output_gm,
  GM_ADDR workspace, GM_ADDR tiling) {
// DT_FLOAT = 0,            // float type
// DT_FLOAT16 = 1,          // fp16 type
// DT_BF16 = 27,            // bf16 type
GET_TILING_DATA(tempTilingGm, tiling);
if (tempTilingGm.isDoubleBuffer == 1) {
if (TILING_KEY_IS(1)) {
SwiGluGradSingle <SwiGluGradBF16<half, float, half, 2>, half, half> op;
op.Init(gradout_gm, input_gm, output_gm, tiling);
op.Process();
} else if (TILING_KEY_IS(0)) {
SwiGluGradSingle<SwiGluGradVector<float, float, float, float, float, 2>, float, float> op;
op.Init(gradout_gm, input_gm, output_gm, tiling);
op.Process();
} 
#if defined(__CCE_AICORE__) && __CCE_AICORE__  == 220
else if (TILING_KEY_IS(27)) {
SwiGluGradSingle <SwiGluGradBF16<bfloat16_t, float, bfloat16_t, 2>, bfloat16_t, bfloat16_t> op;
op.Init(gradout_gm, input_gm, output_gm, tiling);
op.Process();
}
#endif
} else {
if (TILING_KEY_IS(1)) {
SwiGluGradSingle <SwiGluGradBF16<half, float, half, 1>, half, half> op;
op.Init(gradout_gm, input_gm, output_gm, tiling);
op.Process();
} else if (TILING_KEY_IS(0)) {
SwiGluGradSingle<SwiGluGradVector<float, float, float, float, float, 1>, float, float> op;
op.Init(gradout_gm, input_gm, output_gm, tiling);
op.Process();
} 
#if defined(__CCE_AICORE__) && __CCE_AICORE__  == 220   
else if (TILING_KEY_IS(27)) {
SwiGluGradSingle <SwiGluGradBF16<bfloat16_t, float, bfloat16_t, 1>, bfloat16_t, bfloat16_t> op;
op.Init(gradout_gm, input_gm, output_gm, tiling);
op.Process();
}
#endif
}
}