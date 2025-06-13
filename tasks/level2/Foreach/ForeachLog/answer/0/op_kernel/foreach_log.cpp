#include <iostream>
#include "kernel_operator.h"
#include "lib/math/kernel_operator_asin_intf.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint8_t INPUT_PARAMETER_COUNT = 2;
constexpr bool NEED_COPY_OUT = true;
constexpr uint8_t COPY_SPACE_MULTIPLE = 9;


template <typename T>
class KernelForeachBase {
protected:
    __aicore__ inline KernelForeachBase() {};

    __aicore__ inline void Init(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr);

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b) {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };

protected:
    TPipe pipe;

    int64_t blockIdx = 0;

    // tiling params
    uint64_t inputsTensorUbSize = 0;
    const int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = 0;
    uint16_t tensorEnd = 0;
    int64_t tensorStartOffset = 0;
    int64_t tensorEndOffset = 0;

    uint64_t totalTensorUbSize = 0;
    uint32_t maxDataCount = 0;
    uint32_t maxCastDataCount = 0;
};

template <typename T>
__aicore__ inline void KernelForeachBase<T>::Init(
    const ForeachCommonTilingData* tilingData) {
    blockIdx = GetBlockIdx();

    ParseTilingData(tilingData);

    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
            maxDataCount = totalTensorUbSize / sizeof(T);
            maxCastDataCount = inputsTensorUbSize / sizeof(float);
        } else {
            maxDataCount = inputsTensorUbSize / sizeof(T);
        }
    #else
        maxDataCount = inputsTensorUbSize / sizeof(T);
    #endif
}

template <typename T>
__aicore__ inline void KernelForeachBase<T>::ParseTilingData(
    const ForeachCommonTilingData* tilingData) {
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T>
__aicore__ inline __gm__ T* KernelForeachBase<T>::GetTensorAddr(uint16_t index, GM_ADDR tensorPtr) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

template <typename T, typename Predicate, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needCopyOut=NEED_COPY_OUT>
class KernelForeachUnary : public KernelForeachBase<T> {
protected:
    using Base = KernelForeachBase<T>;

    explicit __aicore__ inline KernelForeachUnary(Predicate &p): Base(), pred(p) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor);
private:
    __aicore__ inline void CopyIn(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder);
    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void BeforeProcess();
    __aicore__ inline void AfterProcess();
    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart);

protected:
    TQue<QuePosition::VECIN, bufferNum> dataQueue;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    TQue<QuePosition::VECIN, 1> float32Queue;

private:
    Predicate &pred;
};

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
    const ForeachCommonTilingData* tilingData) {
    Base::Init(tilingData);

    inTensorsPtr = x;
    outTensorsPtr = y;

    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            Base::pipe.InitBuffer(dataQueue, bufferNum, Base::totalTensorUbSize);
            if (needCopyOut) {
                Base::pipe.InitBuffer(outQueue, bufferNum, Base::totalTensorUbSize);
            }
            Base::pipe.InitBuffer(float32Queue, 1, Base::inputsTensorUbSize * paramsCount);
            LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
            float32Queue.EnQue(float32Tensor);
        } else {
            Base::pipe.InitBuffer(dataQueue, bufferNum, Base::inputsTensorUbSize);
            if (needCopyOut) {
                Base::pipe.InitBuffer(outQueue, bufferNum, Base::inputsTensorUbSize);
            }
        }
    #else
        Base::pipe.InitBuffer(dataQueue, bufferNum, Base::inputsTensorUbSize);
        if (needCopyOut) {
            Base::pipe.InitBuffer(outQueue, bufferNum, Base::inputsTensorUbSize);
        }
    #endif
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::Process() {
    /*将中间量预留出来*/
    LocalTensor<float> float32Tensor;
    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            float32Tensor = float32Queue.DeQue<float>();
        }
    #endif

    BeforeProcess();
    for (uint16_t i = Base::tensorStart; i <= Base::tensorEnd; i++) {
        int64_t cursorStart = 0;
        int64_t cursorEnd = Base::tensorDataCountList[i] - 1;
        int64_t dataCount = 0;
        if (i == Base::tensorStart) {
            cursorStart = Base::tensorStartOffset;
        }
        if (i == Base::tensorEnd) {
            cursorEnd = Base::tensorEndOffset;
        }

        dataCount = cursorEnd - cursorStart + 1;

        inTensorsGM.SetGlobalBuffer(Base::GetTensorAddr(i, inTensorsPtr) + cursorStart);
        outTensorsGM.SetGlobalBuffer(Base::GetTensorAddr(i, outTensorsPtr) + cursorStart);
        ProcessPlusInLoop(i, cursorStart);
        SingleTensorProcess(dataCount, float32Tensor);
    }

    AfterProcess();

    #if __CCE_AICORE__ == 220
        if (std::is_same_v<T, bfloat16_t>) {
            float32Queue.FreeTensor(float32Tensor);
        }
    #endif
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::SingleTensorProcess(
    int64_t dataCount, LocalTensor<float> &float32Tensor) {
    // Batch handling and calculation.
    uint32_t copyTimes = dataCount / Base::maxDataCount;
    uint32_t copyTimesRemainder = dataCount % Base::maxDataCount;
    uint32_t tempDataCount = Base::maxDataCount;

    if (copyTimesRemainder > 0) {
        copyTimes++;
    }

    for (uint32_t i = 0; i < copyTimes; i++) {
        bool isRemainder = false;
        if (i == copyTimes - 1 && copyTimesRemainder > 0) {
            isRemainder = true;
            tempDataCount = copyTimesRemainder;
        }
        CopyIn(i, tempDataCount, isRemainder);
        CopyInPlus(i, tempDataCount, isRemainder);
        Compute(i, tempDataCount, float32Tensor, isRemainder);
        if (needCopyOut) {
            CopyOut(i, tempDataCount, isRemainder);
        }
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::CopyIn(uint32_t index, int64_t dataCount, bool isRemainder) {
    LocalTensor<T> dataLocal = dataQueue.template AllocTensor<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorsGM[index * Base::maxDataCount], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorsGM[index * Base::maxDataCount], dataCount);
    }
    dataQueue.EnQue(dataLocal);
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline bool KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
    static_assert(std::is_member_function_pointer_v<decltype(&Predicate::CopyOut)>);
    if (!pred.CopyOut(index, dataCount, isRemainder)) {
        LocalTensor<T> outLocal = outQueue.template DeQue<T>();
	    if (isRemainder) {
	        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
	        DataCopyPad(outTensorsGM[index * Base::maxDataCount], outLocal, copyParams);
	    } else {
	        DataCopy(outTensorsGM[index * Base::maxDataCount], outLocal, dataCount);
	    }

        outQueue.FreeTensor(outLocal);
    }
    return true;
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::Compute(
    uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
    static_assert(std::is_member_function_pointer_v<decltype(&Predicate::Compute)>);
    pred.Compute(index, dataCount, float32Tensor, isRemainder);
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {
    if (std::is_member_function_pointer_v<decltype(&Predicate::CopyInPlus)>) {
        pred.CopyInPlus(index, dataCount, isRemainder);
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::BeforeProcess(){
    if (std::is_member_function_pointer_v<decltype(&Predicate::BeforeProcess)>) {
        pred.BeforeProcess();
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::AfterProcess(){
    if (std::is_member_function_pointer_v<decltype(&Predicate::AfterProcess)>) {
        pred.AfterProcess();
    }
}

template <typename T, typename Predicate, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void KernelForeachUnary<T, Predicate, bufferNum, paramsCount, needCopyOut>::ProcessPlusInLoop(uint32_t index, uint64_t cursorStart){
    if (std::is_member_function_pointer_v<decltype(&Predicate::ProcessPlusInLoop)>) {
        pred.ProcessPlusInLoop(index, cursorStart);
    }
}


template <typename T>
using ImplictOutputOp = void (const LocalTensor<T>&, const LocalTensor<T>&, const int32_t&);

template <typename T, typename P, ImplictOutputOp<P> *op, uint8_t paramsCount>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        LocalTensor<T> &dataLocal,
        LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount,
        int64_t dataCount) {
        op(dataLocal, dataLocal, dataCount);
    }
};


#if __CCE_AICORE__ == 220
    template <ImplictOutputOp<float> *op, uint8_t paramsCount>
    class InnerComputer<bfloat16_t, float, op, paramsCount> {
    public:
        __aicore__ inline void Compute(
            LocalTensor<bfloat16_t> &dataLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount,
            int64_t dataCount) {
            uint32_t castTimes = dataCount / maxCastDataCount;
            uint32_t castTimesRemainder = dataCount % maxCastDataCount;

            for (uint32_t i = 0; i < castTimes; i++) {
                ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, i, maxCastDataCount);
            }

            if (castTimesRemainder > 0) {
                ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, castTimes, castTimesRemainder);
            }
        }

    private:
        __aicore__ inline void ComputePerCast(
            LocalTensor<bfloat16_t> &dataLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount, uint32_t index, int64_t dataCount) {
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
            uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
            op(float32Tensor[offset], float32Tensor, dataCount);
            PipeBarrier<PIPE_V>();
            Cast(dataLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
            PipeBarrier<PIPE_V>();
        }
    };
#endif

template <typename T, typename P, ImplictOutputOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT>
class ForeachImplictOutput : public KernelForeachUnary<T, ForeachImplictOutput<T, P, op, bufferNum, paramsCount>, bufferNum, paramsCount, false> {
public:
    using Base = KernelForeachUnary<T, ForeachImplictOutput<T, P, op, bufferNum, paramsCount>, bufferNum, paramsCount, false>;
    using Operator = ImplictOutputOp<P>;

    __aicore__ inline ForeachImplictOutput() : Base(*this) {};
    using Base::Init;
    using Base::Process;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();
        InnerComputer<T, P, op, paramsCount> computer;
        computer.Compute(
            dataLocal,
            float32Tensor,
            Base::maxCastDataCount,
            dataCount);

        // Transport can be performed only after the Muls is complete.
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(Base::outTensorsGM[index * Base::maxDataCount], dataLocal, copyParams);
        } else {
            DataCopy(Base::outTensorsGM[index * Base::maxDataCount], dataLocal, dataCount);
        }
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

        Base::dataQueue.FreeTensor(dataLocal);
    }

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
        return false;
    }

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

template <typename T>
__aicore__ void LogAdapter(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& uValue) {
    Log<T>(dstLocal, srcLocal);
}

// 核函数入口
extern "C" __global__ __aicore__ void foreach_log(GM_ADDR x,  GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    // 参数说明：
    //   x (aclTensorList*，输入)：公式中的 X，Device 侧的 aclTensorList，支持数据类型 BFLOAT16、FLOAT16、FLOAT，
    //                             维度不超过 8 维，数据格式为 ND。
    //   y (aclTensorList*，输出)：公式中的 Y，Device 侧的 aclTensorList，数据类型、格式和 shape 与 x 一致，
    //                             支持 BFLOAT16、FLOAT16、FLOAT，维度不超过 8 维，数据格式为 ND。
    //   workspace：Device 侧的工作空间地址。
    //   tiling：tiling结构体在device侧的首地址。
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachImplictOutput<half, half, LogAdapter<half>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachImplictOutput<float, float, LogAdapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        ForeachImplictOutput<bfloat16_t, float, LogAdapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
}