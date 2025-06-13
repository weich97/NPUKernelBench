#include <iostream>
#include "kernel_operator.h"
#include "lib/math/kernel_operator_asin_intf.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint8_t INPUT_PARAMETER_COUNT = 2;
constexpr bool NEED_COPY_OUT = true;
constexpr uint8_t COPY_SPACE_MULTIPLE = 9;
constexpr int16_t MAX_REPEATS = 255;
constexpr int16_t BYTES_PER_REPEAT = 256;
constexpr int8_t BYTES_PER_BLOCK = 32;
constexpr int8_t STRIDES_PER_REPEAT = 8;


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
using NoScalarBinaryOp = void (const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, const int32_t&);

template <typename T, typename P, NoScalarBinaryOp<P> *op>
class InnerComputer {
public:
    __aicore__ inline void Compute(
        LocalTensor<T> &inLocal_1,
        LocalTensor<T> &inLocal_2,
        const LocalTensor<T> &outLocal,
        LocalTensor<float> &float32Tensor,
        uint32_t maxCastDataCount,
        int64_t dataCount) {
        PipeBarrier<PIPE_V>();
        op(outLocal, inLocal_1, inLocal_2, dataCount);
        PipeBarrier<PIPE_V>();
    }
};

#if __CCE_AICORE__ == 220
    template <NoScalarBinaryOp<float> *op>
    class InnerComputer<bfloat16_t, float, op> {
    public:
        __aicore__ inline void Compute(
            LocalTensor<bfloat16_t> &inLocal_1,
            LocalTensor<bfloat16_t> &inLocal_2,
            const LocalTensor<bfloat16_t> &outLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount,
            int64_t dataCount) {
            uint32_t castTimes = dataCount / maxCastDataCount;
            uint32_t castTimesRemainder = dataCount % maxCastDataCount;
            for (uint32_t i = 0; i < castTimes; i++) {
                ComputePerCast(
                    inLocal_1, inLocal_2, outLocal, float32Tensor,
                    maxCastDataCount, i, maxCastDataCount);
            }

            if (castTimesRemainder > 0) {
                ComputePerCast(
                    inLocal_1, inLocal_2, outLocal, float32Tensor,
                    maxCastDataCount, castTimes, castTimesRemainder);
            }
        }

    private:
        __aicore__ inline void ComputePerCast(
            LocalTensor<bfloat16_t> &inLocal_1,
            LocalTensor<bfloat16_t> &inLocal_2,
            const LocalTensor<bfloat16_t> &outLocal,
            LocalTensor<float> &float32Tensor,
            uint32_t maxCastDataCount, uint32_t index, int64_t dataCount) {
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor, inLocal_1[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor[maxCastDataCount], inLocal_2[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
            op(float32Tensor, float32Tensor, float32Tensor[maxCastDataCount], dataCount);
            PipeBarrier<PIPE_V>();
            Cast(outLocal[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, dataCount);
        }
    };
#endif

template <typename T, typename P, NoScalarBinaryOp<P> *op, int32_t bufferNum=BUFFER_NUM, uint8_t paramsCount=INPUT_PARAMETER_COUNT, bool needCopyOut=NEED_COPY_OUT>
class ForeachNoScalarBinary : public KernelForeachUnary<T, ForeachNoScalarBinary<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut> {
public:
    using Base = KernelForeachUnary<T, ForeachNoScalarBinary<T, P, op, bufferNum, paramsCount, needCopyOut>, bufferNum, paramsCount, needCopyOut>;
    using Operator = NoScalarBinaryOp<P>;
    __aicore__ inline void Init(GM_ADDR inputs_1, GM_ADDR inputs_2, GM_ADDR y, GM_ADDR workspace,
                            const ForeachCommonTilingData* tilingData);
    __aicore__ inline ForeachNoScalarBinary() : Base(*this) {};
    using Base::Process;

protected:
    TQue<QuePosition::VECIN, BUFFER_NUM> InQueue_2;
    GlobalTensor<T> inTensorsGM_2;
    GM_ADDR inTensorsPtr_2 = nullptr;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float> &float32Tensor, bool isRemainder) {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();
        LocalTensor<T> inLocal_2 = InQueue_2.DeQue<T>();
        LocalTensor<T> outLocal = Base::outQueue.template AllocTensor<T>();

        InnerComputer<T, P, op> computer;
        computer.Compute(
            dataLocal,
            inLocal_2,
            outLocal,
            float32Tensor,
            Base::maxCastDataCount,
            dataCount);

        Base::dataQueue.FreeTensor(dataLocal);
        InQueue_2.FreeTensor(inLocal_2);
        Base::outQueue.template EnQue<T>(outLocal);
    }

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {
        LocalTensor<T> inLocal_2 = InQueue_2.AllocTensor<T>();
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(inLocal_2, inTensorsGM_2[index * Base::maxDataCount], copyParams, padParams);
        } else {
            DataCopy(inLocal_2, inTensorsGM_2[index * Base::maxDataCount], dataCount);
        }
        InQueue_2.EnQue(inLocal_2);
    }

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) {
        return false;
    }

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {
        inTensorsGM_2.SetGlobalBuffer(Base::GetTensorAddr(index, inTensorsPtr_2) + cursorStart);
    }

    friend Base;
};

template <typename T, typename P, NoScalarBinaryOp<P> *op, int32_t bufferNum, uint8_t paramsCount, bool needCopyOut>
__aicore__ inline void ForeachNoScalarBinary<T, P, op, bufferNum, paramsCount, needCopyOut>::Init(GM_ADDR inputs_1, GM_ADDR inputs_2, GM_ADDR y, GM_ADDR workspace,
                            const ForeachCommonTilingData* tilingData) {
    Base::Init(inputs_1, y, workspace, tilingData);
    inTensorsPtr_2 = inputs_2;
    #if __CCE_AICORE__ == 220
        if (std::is_same<T, bfloat16_t>::value) {
            uint64_t totalTensorUbSize = Base::inputsTensorUbSize * COPY_SPACE_MULTIPLE;
            Base::pipe.InitBuffer(InQueue_2, bufferNum, totalTensorUbSize);
        } else {
            Base::pipe.InitBuffer(InQueue_2, bufferNum, Base::inputsTensorUbSize);
        }
    #else
        Base::pipe.InitBuffer(InQueue_2, bufferNum, Base::inputsTensorUbSize);
    #endif
}

// 核函数入口
extern "C" __global__ __aicore__ void foreach_mul_list(GM_ADDR inputs_1, GM_ADDR inputs_2,
    GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
    // 参数说明：
    //   inputs_1 (aclTensorList*，输入)：公式中的 x1，Device 侧的 aclTensorList，支持数据类型 BFLOAT16、FLOAT16、FLOAT、Int32，
    //                             维度不超过 8 维，数据格式为 ND。
    //   inputs_2 (aclTensorList*，输入)：公式中的 x2，Device 侧的 aclTensorList，支持数据类型 BFLOAT16、FLOAT16、FLOAT、Int32，
    //                             维度不超过 8 维，数据格式为 ND。
    //   outputs (aclTensorList*，输出)：公式中的 y，Device 侧的 aclTensorList，数据类型、格式和 shape 与 x1、x2 一致，
    //                             支持 BFLOAT16、FLOAT16、FLOAT、Int32，维度不超过 8 维，数据格式为 ND。
    //   workspace：Device 侧的工作空间地址。
    //   tiling：tiling结构体在device侧的首地址。
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachNoScalarBinary<half, half, Mul> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachNoScalarBinary<float, float, Mul> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(3)) {
        ForeachNoScalarBinary<int, int, Mul> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    }
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(4)) {
        ForeachNoScalarBinary<bfloat16_t, float, Mul> op;
        op.Init(inputs_1, inputs_2, outputs, userWS, &tilingData);
        op.Process();
    }
#endif
}
