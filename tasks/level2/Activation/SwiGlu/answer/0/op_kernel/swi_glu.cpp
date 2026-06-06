#include "kernel_operator.h"

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

using namespace AscendC;
template<typename ParentClass, typename inType, typename outType>
class SwigluSingle : public ParentClass {
public:
    __aicore__ inline SwigluSingle() = default;
    __aicore__ inline ~SwigluSingle() = default;

    __aicore__ inline void Init(GM_ADDR input_gm, GM_ADDR beta_gm, GM_ADDR output_gm, GM_ADDR tiling_gm)
    {
        singleTiling.GetTilingAndOffset(tiling_gm, sizeof(inType));
        InitGmBuffer(input_gm, beta_gm, output_gm);
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
    __aicore__ inline void InitGmBuffer(GM_ADDR input_gm, GM_ADDR beta_gm, GM_ADDR output_gm)
    {
        this->beta = -1.0f /* * (((__gm__ float*)beta_gm)[0]) */;
        // get start index for current core, core parallel
        this->aGm.SetGlobalBuffer((__gm__ inType*)input_gm, singleTiling.totalBlockLen);
        this->cGm.SetGlobalBuffer((__gm__ outType*)output_gm, singleTiling.totalBlockLen / 2);
    }

    __aicore__ inline void CopyIn(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyinParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                            SwiGluCopyParam.splitVecCopyParam.blockLen,
                                            SwiGluCopyParam.splitVecCopyParam.stride,
                                            0};
        // Copy A
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopy(aLocal, this->aGm[offsetParam.splitVecGmOffset1], splitCopyinParams);
        this->inQueueA.EnQue(aLocal);
        // Copy B
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopy(bLocal, this->aGm[offsetParam.splitVecGmOffset2], splitCopyinParams);
        this->inQueueB.EnQue(bLocal);
    }

    __aicore__ inline void CopyOut(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams indepCopyinParams = {SwiGluCopyParam.indepVecCopyParam.blockCount,
                                            SwiGluCopyParam.indepVecCopyParam.blockLen,
                                            0,
                                            SwiGluCopyParam.indepVecCopyParam.stride};
        // deque output tensor from VECOUT queue
        LocalTensor<outType> cLocal = this->outQueueC.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(this->cGm[offsetParam.indepVecGmoffset], cLocal, indepCopyinParams);
        // free output tensor for reuse
        this->outQueueC.FreeTensor(cLocal);
    }

    __aicore__ inline void CopyIn_Non32BAligned(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyinParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                            SwiGluCopyParam.splitVecCopyParam.blockLen,
                                            SwiGluCopyParam.splitVecCopyParam.stride,
                                            0};
        DataCopyPadParams copyPadParams = {false, 0, 0, 0};
        // Copy A
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopyPad(aLocal, this->aGm[offsetParam.splitVecGmOffset1], splitCopyinParams, copyPadParams);
        this->inQueueA.EnQue(aLocal);
        // Copy B
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopyPad(bLocal, this->aGm[offsetParam.splitVecGmOffset2], splitCopyinParams, copyPadParams);
        this->inQueueB.EnQue(bLocal);
    }

    __aicore__ inline void CopyOut_Non32BAligned(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams indepCopyinParams = {SwiGluCopyParam.indepVecCopyParam.blockCount,
                                            SwiGluCopyParam.indepVecCopyParam.blockLen,
                                            0,
                                            SwiGluCopyParam.indepVecCopyParam.stride};
        // deque output tensor from VECOUT queue
        LocalTensor<outType> cLocal = this->outQueueC.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopyPad(this->cGm[offsetParam.indepVecGmoffset], cLocal, indepCopyinParams);
        // free output tensor for reuse
        this->outQueueC.FreeTensor(cLocal);
    }

private:
    SwigluSingleTilingKernel singleTiling;
};

using namespace AscendC;
template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
class SwigluVectorBF16 {
public:
    __aicore__ inline SwigluVectorBF16() {}
    __aicore__ inline ~SwigluVectorBF16() {}

protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);

    float beta = 1.0;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECOUT, bufferNum> outQueueC;

    TBuf<TPosition::VECCALC> inputTempBuffer;
    TBuf<TPosition::VECCALC> outputTempBuffer; // Implementation note.

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<outType> cGm;
};

  template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
  __aicore__ inline void SwigluVectorBF16<inType, calcType, outType, bufferNum>::InitUbBuffer(uint64_t tileLength) {
      // pipe alloc memory to queue, the unit is Bytes
      pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(inType));
      pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(inType));
      pipe.InitBuffer(outQueueC, bufferNum, tileLength * sizeof(outType));

      pipe.InitBuffer(inputTempBuffer, tileLength * sizeof(calcType));
      pipe.InitBuffer(outputTempBuffer, tileLength * sizeof(calcType));
  }

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVectorBF16<inType, calcType, outType, bufferNum>::Compute(uint64_t curTileLen)
{
    LocalTensor<inType> aLocal_ = inQueueA.template DeQue<inType>();
    LocalTensor<outType> cLocal_ = outQueueC.template AllocTensor<outType>();

    LocalTensor<calcType> aLocal = inputTempBuffer.Get<calcType>();
    LocalTensor<calcType> cLocal = outputTempBuffer.Get<calcType>();
    Cast(aLocal, aLocal_, RoundMode::CAST_NONE, curTileLen);
    pipe_barrier(PIPE_V);
    inQueueA.template FreeTensor(aLocal_);

    Muls(cLocal, aLocal, beta, curTileLen);
    pipe_barrier(PIPE_V);
    Exp(cLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);
    Adds(cLocal, cLocal, calcType(1.0), curTileLen);
    pipe_barrier(PIPE_V);

    Div(cLocal, aLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);

    LocalTensor<inType> bLocal_ = inQueueB.template DeQue<inType>();

    LocalTensor<calcType> bLocal = inputTempBuffer.Get<calcType>();
    Cast(bLocal, bLocal_, RoundMode::CAST_NONE, curTileLen);
    pipe_barrier(PIPE_V);
    inQueueB.template FreeTensor(bLocal_);

    Mul(cLocal, cLocal, bLocal, curTileLen);
    pipe_barrier(PIPE_V);

    Cast(cLocal_, cLocal, RoundMode::CAST_RINT, curTileLen);
    pipe_barrier(PIPE_V);
    // enque the output tensor to VECOUT queue
    outQueueC.template EnQue<outType>(cLocal_);
    // free input tensors for reuse
}

using namespace AscendC;
template<typename inType, typename outType, uint16_t bufferNum>
class SwigluVector {
  public:
    __aicore__ inline SwigluVector() {}
    __aicore__ inline ~SwigluVector() {}
  protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);
    float beta = 1.0;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECOUT, bufferNum> outQueueC;
    TBuf<TPosition::VECCALC> tmpQueue;

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<outType> cGm;
};

template<typename inType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVector<inType, outType, bufferNum>::InitUbBuffer(uint64_t tileLength) {
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(outQueueC, bufferNum, tileLength * sizeof(outType));
    pipe.InitBuffer(tmpQueue, tileLength * sizeof(float));
    LocalTensor<float> tempLocal = tmpQueue.Get<float>();
    Duplicate<float>(tempLocal, (float)(1.0), tileLength);
}

template<typename inType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVector<inType, outType, bufferNum>::Compute(uint64_t curTileLen)
{
    LocalTensor<inType> aLocal = inQueueA.template DeQue<inType>();
    LocalTensor<outType> cLocal = outQueueC.template AllocTensor<outType>();
    pipe_barrier(PIPE_V);
    Muls(cLocal, aLocal, beta, curTileLen);
    pipe_barrier(PIPE_V);
    Exp(cLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);
    Adds(cLocal, cLocal, (outType)(1.0), curTileLen);
    pipe_barrier(PIPE_V);

    LocalTensor<float> tempLocal = tmpQueue.Get<float>();
    pipe_barrier(PIPE_V);
    Div(cLocal, tempLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);
    Mul(cLocal, cLocal, aLocal, curTileLen);
    pipe_barrier(PIPE_V);
    inQueueA.template FreeTensor(aLocal);

    LocalTensor<inType> bLocal = inQueueB.template DeQue<inType>();
    pipe_barrier(PIPE_V);
    Mul(cLocal, cLocal, bLocal, curTileLen);
    pipe_barrier(PIPE_V);
    inQueueB.template FreeTensor(bLocal);
    // enque the output tensor to VECOUT queue
    outQueueC.template EnQue<outType>(cLocal);
    // free input tensors for reuse
}

using namespace AscendC;
extern "C" __global__ __aicore__ void swi_glu(GM_ADDR input_gm, GM_ADDR output_gm,
                                              GM_ADDR workspace, GM_ADDR tiling) {
  // Implementation note.
  // Implementation note.
  // Implementation note.

  // DT_FLOAT = 0,            // float type
  // DT_FLOAT16 = 1,          // fp16 type
  // DT_BF16 = 27,            // bf16 type
  GET_TILING_DATA(tempTilingGm, tiling);
  if (tempTilingGm.isDoubleBuffer == 1) {
    if (TILING_KEY_IS(1)) {
      SwigluSingle<SwigluVectorBF16<half, float, half, 2>, half, half> op;
      op.Init(input_gm, nullptr, output_gm, tiling);
      op.Process();
    } else if (TILING_KEY_IS(0)) {
      SwigluSingle<SwigluVector<float, float, 2>, float, float> op;
      op.Init(input_gm, nullptr, output_gm, tiling);
      op.Process();
    } 
#if defined(__CCE_AICORE__) && __CCE_AICORE__  == 220
    else if (TILING_KEY_IS(27)) {
      SwigluSingle<SwigluVectorBF16<bfloat16_t, float, bfloat16_t, 2>, bfloat16_t, bfloat16_t> op;
      op.Init(input_gm, nullptr, output_gm, tiling);
      op.Process();
    }
#endif
  } else {
    if (TILING_KEY_IS(1)) {
      SwigluSingle<SwigluVectorBF16<half, float, half, 1>, half, half> op;
      op.Init(input_gm, nullptr, output_gm, tiling);
      op.Process();
    } else if (TILING_KEY_IS(0)) {
      SwigluSingle<SwigluVector<float, float, 1>, float, float> op;
      op.Init(input_gm, nullptr, output_gm, tiling);
      op.Process();
    } 
#if defined(__CCE_AICORE__) && __CCE_AICORE__  == 220
    else if (TILING_KEY_IS(27)) {
      SwigluSingle<SwigluVectorBF16<bfloat16_t, float, bfloat16_t, 1>, bfloat16_t, bfloat16_t> op;
      op.Init(input_gm, nullptr, output_gm, tiling);
      op.Process();
    }
#endif
  }
}