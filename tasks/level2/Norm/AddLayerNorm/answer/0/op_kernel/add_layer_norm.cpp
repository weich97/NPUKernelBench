#include "kernel_operator.h"
#if __CCE_AICORE__ == 220
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#endif

using namespace AscendC;
static constexpr float ZERO = 0;
constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr int32_t ELEM_PER_REP_FP32 = 64;  // ONE_REPEAT_BYTE_SIZE / sizeof(float)
constexpr int32_t ELEM_PER_REP_FP16 = 128;
constexpr uint32_t MAX_REP_NUM = 255;
constexpr uint32_t BROADCAST_ND_DIM_NUM = 2;     // only support 1 or 2
constexpr uint32_t BROADCAST_ND_LAST_INDEX = 1;  // only support 0 or 1

#if __CCE_AICORE__ == 220
#define OUTPUT_MEAN_RSTD 1
#define SUPPORT_BF16 1
#else
#define OUTPUT_MEAN_RSTD 0
#define SUPPORT_BF16 0
#endif

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using true_type = IntegralConstant<bool, true>;
using false_type = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public false_type {};
template <typename Tp>
struct IsSame<Tp, Tp> : public true_type {};

template <typename T, template <typename U> typename R, template <typename U> typename S>
__aicore__ inline void DataCopyEx(const R<T> &dst, const S<T> &src, const uint32_t len, const uint32_t count = 1,
    const DataCopyPadParams &padParams = {})
{
#if __CCE_AICORE__ == 220
    DataCopyParams copyParams;
    copyParams.blockLen = len * sizeof(T);
    copyParams.blockCount = count;
    if constexpr (IsSame<R<T>, AscendC::LocalTensor<T>>::value) {
        DataCopyPad(dst, src, copyParams, padParams);
    } else {
        DataCopyPad(dst, src, copyParams);
    }
#else
    auto elementCount = len * count;
    int32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
    if (elementCount % numPerBlock == 0) {
        DataCopy(dst, src, elementCount);
    } else {
        if constexpr (IsSame<R<T>, AscendC::LocalTensor<T>>::value) {
            auto num = AlignUp(elementCount, numPerBlock);
            DataCopy(dst, src, num);
        } else {
            int32_t num = elementCount / numPerBlock * numPerBlock;
            DataCopy(dst, src, num);
            if (elementCount != num) {
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                for (int32_t i = 0; i < numPerBlock; i++) {
                    auto tensorValue = src.GetValue(elementCount - numPerBlock + i);
                    src.SetValue(i, tensorValue);
                }
                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
                DataCopy(dst[elementCount - numPerBlock], src, numPerBlock);
            }
        }
    }
#endif
}

/*
 * only support count <= 255 * 64 = 16320
 */
__aicore__ inline float ReduceSumFP32(const LocalTensor<float> &src_local, int32_t count)
{
    int32_t elementNumPerRep = ELEM_PER_REP_FP32;
    int32_t repeatTimes = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep;
    int32_t bodyCount = repeatTimes * elementNumPerRep;
#ifdef __CCE_KT_TEST__
    assert(count <= MAX_REPEAT_TIMES * elementNumPerRep);
#endif
    float value = 0.0;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        if (likely(repeatTimes > 0)) {
            AscendCUtils::SetMask<float>(elementNumPerRep);
            vcadd(nullptr, (__ubuf__ float *)src_local.GetPhyAddr(), repeatTimes, 1, 1, 8, true);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
#ifdef __CCE_KT_TEST__
            uint64_t acc_val = get_acc_val();
#else
            uint64_t acc_val = GetAccVal();
#endif
            value = *reinterpret_cast<float *>(&acc_val);
        }
        if (unlikely(tailCount != 0)) {
            AscendCUtils::SetMask<float>(tailCount);
            vcadd(nullptr, (__ubuf__ float *)src_local[bodyCount].GetPhyAddr(), 1, 1, 1, 8, true);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
#ifdef __CCE_KT_TEST__
            uint64_t acc_val = get_acc_val();
#else
            uint64_t acc_val = GetAccVal();
#endif
            value += *reinterpret_cast<float *>(&acc_val);
        }
    }
#else
    ReduceSum(src_local, src_local, src_local, count);
    value = src_local.GetValue(0);
#endif
    return value;
}

__aicore__ inline void ReduceSumShort(const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local,
    const LocalTensor<float> &tmp_local, int32_t align_len, int32_t data_len, int32_t repeat)
{
    int32_t elementNum = ONE_BLK_SIZE / sizeof(float);
    int32_t maxRepeat = ELEM_PER_REP_FP32;
    int32_t tailCount = data_len % elementNum;
    uint32_t index = 0;
    uint8_t repStride = align_len / ONE_BLK_FLOAT_NUM;

    int32_t repeatTimes = repeat / elementNum;
    int32_t bodyCount = repeatTimes * elementNum;
    int32_t repeatTail = repeat % elementNum * elementNum;

    Duplicate<float>(tmp_local, ZERO, repeat * elementNum);
    pipe_barrier(PIPE_V);
    for (index = 0; index + elementNum <= data_len; index += elementNum) {
        Add(tmp_local, tmp_local, src_local[index], elementNum, repeat, {1, 1, 1, 1, 1, repStride});
        pipe_barrier(PIPE_V);
    }
    if (unlikely(tailCount != 0)) {
        Add(tmp_local, tmp_local, src_local[index], tailCount, repeat, {1, 1, 1, 1, 1, repStride});
    }
    pipe_barrier(PIPE_V);
    if (repeatTimes != 0) {
        BlockReduceSum<float>(dst_local, tmp_local, repeatTimes, maxRepeat, 1, 1, elementNum);
    }
    if (repeatTail != 0) {
        BlockReduceSum<float>(dst_local[bodyCount], tmp_local[bodyCount * elementNum], 1, repeatTail, 1, 1, elementNum);
    }
}

__aicore__ inline void ReduceSumForSmallReduceDimPreRepeat(const LocalTensor<float> &dstLocal,
    const LocalTensor<float> &srcLocal, const LocalTensor<float> &tmpLocal, const uint32_t elemNum,
    const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat, const uint8_t repStride)
{
    uint32_t elemIndex = 0;
    for (; elemIndex + ELEM_PER_REP_FP32 <= numLastDim; elemIndex += ELEM_PER_REP_FP32) {
        Add(tmpLocal,
            srcLocal[elemIndex],
            tmpLocal,
            elemNum,
            repeat,
            {1, 1, 1, FLOAT_BLOCK_ELEM, repStride, FLOAT_BLOCK_ELEM});
        pipe_barrier(PIPE_V);
    }
    if (unlikely(tailCount != 0)) {
        Add(tmpLocal,
            srcLocal[elemIndex],
            tmpLocal,
            tailCount,
            repeat,
            {1, 1, 1, FLOAT_BLOCK_ELEM, repStride, FLOAT_BLOCK_ELEM});
    }
    pipe_barrier(PIPE_V);
    AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);  // set mask = 64
#if __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        vcadd((__ubuf__ float *)dstLocal.GetPhyAddr(),
            (__ubuf__ float *)tmpLocal.GetPhyAddr(),
            repeat,
            1,
            1,
            FLOAT_BLOCK_ELEM,
            false);
    }
#else
    vcadd((__ubuf__ float *)dstLocal.GetPhyAddr(),
        (__ubuf__ float *)tmpLocal.GetPhyAddr(),
        repeat,
        1,
        1,
        FLOAT_BLOCK_ELEM);
#endif
}

/*
 * reduce dim form (N, D) to (N, 1)
 * this reduce sum is for small reduce dim.
 */
__aicore__ inline void ReduceSumForSmallReduceDim(const LocalTensor<float> &dstLocal,
    const LocalTensor<float> &srcLocal, const LocalTensor<float> &tmpLocal, const uint32_t numLastDimAligned,
    const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat, const uint8_t repStride)
{
    uint32_t repeatTimes = repeat / MAX_REP_NUM;
    if (repeatTimes == 0) {
        ReduceSumForSmallReduceDimPreRepeat(
            dstLocal, srcLocal, tmpLocal, ELEM_PER_REP_FP32, numLastDim, tailCount, repeat, repStride);
    } else {
        uint32_t repTailNum = repeat % MAX_REP_NUM;
        uint32_t repIndex = 0;
        uint32_t repElem;
        for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
            ReduceSumForSmallReduceDimPreRepeat(dstLocal[repIndex],
                srcLocal[repIndex * numLastDimAligned],
                tmpLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32,
                numLastDim,
                tailCount,
                MAX_REP_NUM,
                repStride);
        }
        if (repTailNum != 0) {
            ReduceSumForSmallReduceDimPreRepeat(dstLocal[repIndex],
                srcLocal[repIndex * numLastDimAligned],
                tmpLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32,
                numLastDim,
                tailCount,
                repTailNum,
                repStride);
        }
    }
}

__aicore__ inline void InitVAForTranspose(__ubuf__ half *transposeDstAddr, __ubuf__ half *transposeSrcAddr)
{
    uint64_t va_reg_array_1[8] = {((uint64_t)transposeDstAddr),
        ((uint64_t)(transposeDstAddr + (int64_t)128)),
        ((uint64_t)(transposeDstAddr + (int64_t)256)),
        ((uint64_t)(transposeDstAddr + (int64_t)384)),
        ((uint64_t)(transposeDstAddr + (int64_t)512)),
        ((uint64_t)(transposeDstAddr + (int64_t)640)),
        ((uint64_t)(transposeDstAddr + (int64_t)768)),
        ((uint64_t)(transposeDstAddr + (int64_t)896))};
    set_va_reg_sb(VA0, va_reg_array_1);
    uint64_t va_reg_array_2[8] = {((uint64_t)(transposeDstAddr + (int64_t)1024)),
        ((uint64_t)(transposeDstAddr + (int64_t)1152)),
        ((uint64_t)(transposeDstAddr + (int64_t)1280)),
        ((uint64_t)(transposeDstAddr + (int64_t)1408)),
        ((uint64_t)(transposeDstAddr + (int64_t)1536)),
        ((uint64_t)(transposeDstAddr + (int64_t)1664)),
        ((uint64_t)(transposeDstAddr + (int64_t)1792)),
        ((uint64_t)(transposeDstAddr + (int64_t)1920))};
    set_va_reg_sb(VA1, va_reg_array_2);
    uint64_t va_reg_array_3[8] = {((uint64_t)transposeSrcAddr),
        ((uint64_t)(transposeSrcAddr + (int64_t)16)),
        ((uint64_t)(transposeSrcAddr + (int64_t)256)),
        ((uint64_t)(transposeSrcAddr + (int64_t)272)),
        ((uint64_t)(transposeSrcAddr + (int64_t)512)),
        ((uint64_t)(transposeSrcAddr + (int64_t)528)),
        ((uint64_t)(transposeSrcAddr + (int64_t)768)),
        ((uint64_t)(transposeSrcAddr + (int64_t)784))};
    set_va_reg_sb(VA2, va_reg_array_3);
    uint64_t va_reg_array_4[8] = {((uint64_t)(transposeSrcAddr + (int64_t)1024)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1040)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1280)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1296)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1536)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1552)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1792)),
        ((uint64_t)(transposeSrcAddr + (int64_t)1808))};
    set_va_reg_sb(VA3, va_reg_array_4);
}

/*
 * for reduce dim > 64, need run InitVAForTranspose first.
 */
__aicore__ inline void CCEBroadCastShort(__ubuf__ int16_t *dstAddr, __ubuf__ float *srcAddr,
    __ubuf__ int16_t *transposeDstAddr, __ubuf__ int16_t *transposeSrcAddr, __ubuf__ int16_t *orOffsetINT16Addr,
    const uint32_t forCount, const uint32_t tailCount, const uint32_t repeat, const uint8_t repStride)
{
    set_vector_mask(0x0, 0xffff);
    vector_dup(orOffsetINT16Addr, (int16_t)0, 1, 0, 0, (uint8_t)0, (uint8_t)0);  // all zero
    copy_ubuf_to_ubuf((__ubuf__ float *)transposeSrcAddr, srcAddr, 0, 1, repStride, 0, 0);
    pipe_barrier(PIPE_V);

    vtranspose((__ubuf__ uint16_t *)transposeDstAddr, (__ubuf__ uint16_t *)transposeSrcAddr);
    AscendCUtils::SetMask<half>(ELEM_PER_REP_FP16);
    pipe_barrier(PIPE_V);

    vor(transposeSrcAddr, transposeDstAddr, orOffsetINT16Addr, 8, 1, 1, 0, 16, 0, 0);
    vor(transposeSrcAddr + (int64_t)ELEM_PER_REP_FP16,
        transposeDstAddr + (int64_t)ELEM_PER_REP_FP16,
        orOffsetINT16Addr,
        8,
        1,
        1,
        0,
        16,
        0,
        0);
    pipe_barrier(PIPE_V);
    scatter_vnchwconv_b16(VA0, VA2, 8, 1, 2);  // transpose
    pipe_barrier(PIPE_V);
    for (int64_t forIndex = 0; forIndex < (int64_t)forCount; ++forIndex) {
        vor(dstAddr + (forIndex * (int64_t)ELEM_PER_REP_FP16),
            transposeDstAddr,
            orOffsetINT16Addr,
            repeat,
            1,
            0,
            0,
            repStride,
            1,
            0);
    }
    if (tailCount != 0) {
        AscendCUtils::SetMask<half>(tailCount * 2);
        vor(dstAddr + (forCount * (int64_t)ELEM_PER_REP_FP16),
            transposeDstAddr,
            orOffsetINT16Addr,
            repeat,
            1,
            0,
            0,
            repStride,
            1,
            0);
    }
}

__aicore__ inline void Level0MulFp32Short(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
    const LocalTensor<float> &src1Local, uint32_t alignElem, uint32_t repeat, uint32_t processElem)
{
    uint32_t maxElemFp32 = ELEM_PER_REP_FP32;
    uint8_t repStride = alignElem / FLOAT_BLOCK_ELEM;
    uint32_t tailCount = processElem % maxElemFp32;

    uint32_t repeatTimes = repeat / MAX_REP_NUM;

    uint32_t index = 0;
    uint32_t elemIndex = 0;
    if (likely(repeatTimes == 0)) {
        elemIndex = 0;
        for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
            Mul(dstLocal[elemIndex],
                src0Local[elemIndex],
                src1Local[elemIndex],
                maxElemFp32,
                repeat,
                {1, 1, 1, repStride, 0, repStride});
        }
        if (tailCount != 0) {
            Mul(dstLocal[elemIndex],
                src0Local[elemIndex],
                src1Local[elemIndex],
                tailCount,
                repeat,
                {1, 1, 1, repStride, 0, repStride});
        }
    } else {
        uint32_t repTailNum = repeat % MAX_REP_NUM;
        uint32_t repIndex = 0;
        uint32_t repElem;
        for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
            elemIndex = 0;
            repElem = repIndex * alignElem;
            for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    maxElemFp32,
                    MAX_REP_NUM,
                    {1, 1, 1, repStride, 0, repStride});
            }
            if (tailCount != 0) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    tailCount,
                    MAX_REP_NUM,
                    {1, 1, 1, repStride, 0, repStride});
            }
        }
        if (repTailNum != 0) {
            elemIndex = 0;
            for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    maxElemFp32,
                    repTailNum,
                    {1, 1, 1, repStride, 0, repStride});
            }
            if (tailCount != 0) {
                index = repElem + elemIndex;
                Mul(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    tailCount,
                    repTailNum,
                    {1, 1, 1, repStride, 0, repStride});
            }
        }
    }
}

__aicore__ inline void Level0AddFp32Short(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
    const LocalTensor<float> &src1Local, uint32_t alignElem, uint32_t repeat, uint32_t processElem)
{
    uint32_t maxElemFp32 = ELEM_PER_REP_FP32;
    uint8_t repStride = alignElem / FLOAT_BLOCK_ELEM;
    uint32_t tailCount = processElem % maxElemFp32;

    uint32_t repeatTimes = repeat / MAX_REP_NUM;

    uint32_t index = 0;
    uint32_t elemIndex = 0;
    if (likely(repeatTimes == 0)) {
        elemIndex = 0;
        for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
            Add(dstLocal[elemIndex],
                src0Local[elemIndex],
                src1Local[elemIndex],
                maxElemFp32,
                repeat,
                {1, 1, 1, repStride, 0, repStride});
        }
        if (tailCount != 0) {
            Add(dstLocal[elemIndex],
                src0Local[elemIndex],
                src1Local[elemIndex],
                tailCount,
                repeat,
                {1, 1, 1, repStride, 0, repStride});
        }
    } else {
        uint32_t repTailNum = repeat % MAX_REP_NUM;
        uint32_t repIndex = 0;
        uint32_t repElem;
        for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
            elemIndex = 0;
            repElem = repIndex * alignElem;
            for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                index = repElem + elemIndex;
                Add(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    maxElemFp32,
                    MAX_REP_NUM,
                    {1, 1, 1, repStride, 0, repStride});
            }
            if (tailCount != 0) {
                index = repElem + elemIndex;
                Add(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    tailCount,
                    MAX_REP_NUM,
                    {1, 1, 1, repStride, 0, repStride});
            }
        }
        if (repTailNum != 0) {
            elemIndex = 0;
            for (; elemIndex + maxElemFp32 <= processElem; elemIndex += maxElemFp32) {
                index = repElem + elemIndex;
                Add(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    maxElemFp32,
                    repTailNum,
                    {1, 1, 1, repStride, 0, repStride});
            }
            if (tailCount != 0) {
                index = repElem + elemIndex;
                Add(dstLocal[elemIndex],
                    src0Local[index],
                    src1Local[elemIndex],
                    tailCount,
                    repTailNum,
                    {1, 1, 1, repStride, 0, repStride});
            }
        }
    }
}

template <typename T_X1, typename T_X2, typename T_GAMMA, typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormBetterUB {
#define IS_ADDITIONAL_OUTPUT_ENABLE ((TILING_KEY % 1000) / 100 == 1)
#define IS_BIAS_PRESENT ((TILING_KEY % 10) == 1)
#define IS_BIAS_BROADCAST ((TILING_KEY % 10) == 2)

public:
    __aicore__ inline KernelAddLayerNormBetterUB(TPipe *pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y > 0) {
            return (x + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline uint32_t ROUND_UP32(uint32_t x)
    {
        return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return x < y ? x : y;
    }

    __aicore__ inline uint32_t MAX(uint32_t x, uint32_t y)
    {
        return x > y ? x : y;
    }

    __aicore__ inline void InitVar(uint32_t num_core_, uint32_t num_Last_dim_, uint32_t num_first_dim_,
        uint32_t nl_first_dim_per_core_, uint32_t l_first_dim_per_core_, uint32_t first_dim_per_time_,
        uint32_t last_dim_per_time_, float eps_, float aveNum_, uint32_t col_move_cnt_, uint32_t col_tail_)
    {
        numCore = num_core_;
        numLastDim = num_Last_dim_;
        numFirstDim = num_first_dim_;
        nlFirstDimPerCore = nl_first_dim_per_core_;
        lFirstDimPerCore = l_first_dim_per_core_;
        firstDimPerTime = first_dim_per_time_;
        lastDimPerTime = last_dim_per_time_;
        aveNum = aveNum_;
        eps = eps_;
        colMoveCnt = col_move_cnt_;
        colTail = col_tail_;
    }

    __aicore__ inline void Init(__gm__ uint8_t *x1, __gm__ uint8_t *x2, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
        __gm__ uint8_t *bias, __gm__ uint8_t *y, __gm__ uint8_t *mean, __gm__ uint8_t *rstd, __gm__ uint8_t *x,
        __gm__ uint8_t *workspace, uint32_t num_core_, uint32_t num_Last_dim_, uint32_t num_first_dim_,
        uint32_t nl_first_dim_per_core_, uint32_t l_first_dim_per_core_, uint32_t first_dim_per_time_,
        uint32_t last_dim_per_time_, float eps_, float aveNum_, uint32_t col_move_cnt_, uint32_t col_tail_,
        uint32_t workspace_size)
    {
        InitVar(num_core_,
            num_Last_dim_,
            num_first_dim_,
            nl_first_dim_per_core_,
            l_first_dim_per_core_,
            first_dim_per_time_,
            last_dim_per_time_,
            eps_,
            aveNum_,
            col_move_cnt_,
            col_tail_);
        if (block_idx != numCore - 1) {
            rowWork = nlFirstDimPerCore;
            rowStep = firstDimPerTime;
        } else {
            rowWork = lFirstDimPerCore;
            rowStep = MIN(firstDimPerTime, rowWork);
        }
        rowTail_ = (rowWork % rowStep == 0) ? rowStep : (rowWork % rowStep);
        gmOffset_ = nlFirstDimPerCore * numLastDim;
        x1Gm.SetGlobalBuffer((__gm__ T *)(x1) + block_idx * gmOffset_);
        x2Gm.SetGlobalBuffer((__gm__ T *)(x2) + block_idx * gmOffset_);
        if constexpr (IS_BIAS_PRESENT) {
            biasGm.SetGlobalBuffer((__gm__ T *)(bias) + block_idx * gmOffset_);
        } else if constexpr (IS_BIAS_BROADCAST) {
            biasGm.SetGlobalBuffer((__gm__ T *)bias);
        }
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma);
        betaGm.SetGlobalBuffer((__gm__ T *)beta);
        yGm.SetGlobalBuffer((__gm__ T *)(y) + block_idx * gmOffset_);
        // mean/rstd always output fp32
        meanGm.SetGlobalBuffer((__gm__ float *)mean + block_idx * nlFirstDimPerCore);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + block_idx * nlFirstDimPerCore);
        xGm.SetGlobalBuffer((__gm__ T *)(x) + block_idx * gmOffset_);

        numLastDimAligned = numLastDim;
        if (ROUND_UP32(numLastDim * sizeof(T)) != numLastDim * sizeof(T)) {
            lastDimPad = true;
            numLastDimAligned = ROUND_UP32(numLastDim * sizeof(T)) / sizeof(T);
        }

        Ppipe->InitBuffer(x1x2Que, BUFFER_NUM, ROUND_UP32(2 * rowStep * numLastDimAligned * sizeof(T)));
        Ppipe->InitBuffer(yQue, BUFFER_NUM, ROUND_UP32(rowStep * numLastDimAligned * sizeof(T)));

        Ppipe->InitBuffer(betaBuf, ROUND_UP32(numLastDimAligned * sizeof(float)));
        Ppipe->InitBuffer(gammaBuf, ROUND_UP32(numLastDimAligned * sizeof(float)));

        Ppipe->InitBuffer(xBufFp32, ROUND_UP32(rowStep * numLastDimAligned * sizeof(float)));
        Ppipe->InitBuffer(yBufFp32, ROUND_UP32(rowStep * numLastDimAligned * sizeof(float)));

        if constexpr (IS_BIAS_BROADCAST) {
            Ppipe->InitBuffer(biasBuf, ROUND_UP32(numLastDim * sizeof(T)));
        }

#if OUTPUT_MEAN_RSTD == 1
        Ppipe->InitBuffer(meanQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
        Ppipe->InitBuffer(rstdQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
#endif
    }

    __aicore__ inline void Process()
    {
        int32_t rowMoveCnt = CEIL_DIV(rowWork, rowStep);

        DataCopyPadParams padParams;
        if (lastDimPad) {
            padParams.isPad = true;
            padParams.paddingValue = 0;
            padParams.rightPadding = numLastDimAligned - numLastDim;
        }

        LocalTensor<float> betaLocal = betaBuf.template Get<float>();
        LocalTensor<float> gammaLocal = gammaBuf.template Get<float>();

        if constexpr (IsSame<float, T>::value) {
            DataCopyEx(betaLocal, betaGm, numLastDim);
            DataCopyEx(gammaLocal, gammaGm, numLastDim);
        } else {
            auto betaLocalHalf = betaLocal.ReinterpretCast<T>();
            auto gammaLocalHalf = gammaLocal.ReinterpretCast<T>();
            DataCopyEx(betaLocalHalf[numLastDimAligned], betaGm, numLastDim);
            DataCopyEx(gammaLocalHalf[numLastDimAligned], gammaGm, numLastDim);
        }

        LocalTensor<T> biasLocal;
        if constexpr (IS_BIAS_BROADCAST) {
            biasLocal = biasBuf.template Get<T>();
            DataCopyEx(biasLocal, biasGm, numLastDim);
        }

        uint32_t gmOffset = 0;
        auto elementCount = numLastDimAligned * rowStep;

        {
            LocalTensor<T> x1x2LocalIn = x1x2Que.template AllocTensor<T>();
            DataCopyEx(x1x2LocalIn[0], x1Gm[gmOffset], numLastDim, rowStep, padParams);
            DataCopyEx(x1x2LocalIn[elementCount], x2Gm[gmOffset], numLastDim, rowStep, padParams);
            x1x2Que.EnQue(x1x2LocalIn);
            auto x1x2Local = x1x2Que.template DeQue<T>();

            if constexpr (!IsSame<T, float>::value) {
                Cast(gammaLocal, gammaLocal.ReinterpretCast<T>()[numLastDimAligned], RoundMode::CAST_NONE, numLastDim);
                Cast(betaLocal, betaLocal.ReinterpretCast<T>()[numLastDimAligned], RoundMode::CAST_NONE, numLastDim);
            }

            if constexpr (IS_BIAS_BROADCAST) {
                CopyInAndAddBroadCast(0, rowStep, biasLocal, x1x2Local, elementCount);
            }
            CopyOutAdditionalOutput(0, rowStep);
            precisionCompute(rowStep, gammaLocal, betaLocal, x1x2Local, elementCount);
            CopyOut(0, rowStep);
            gmOffset += rowStep * numLastDim;
        }
        for (int32_t rowIdx = 1; rowIdx < rowMoveCnt - 1; ++rowIdx) {
            LocalTensor<T> x1x2LocalIn = x1x2Que.template AllocTensor<T>();
            DataCopyEx(x1x2LocalIn[0], x1Gm[gmOffset], numLastDim, rowStep, padParams);
            DataCopyEx(x1x2LocalIn[elementCount], x2Gm[gmOffset], numLastDim, rowStep, padParams);
            x1x2Que.EnQue(x1x2LocalIn);
            auto x1x2Local = x1x2Que.template DeQue<T>();

            if constexpr (IS_BIAS_BROADCAST) {
                CopyInAndAddBroadCast(rowIdx, rowStep, biasLocal, x1x2Local, elementCount);
            }
            CopyOutAdditionalOutput(rowIdx, rowStep);
            precisionCompute(rowStep, gammaLocal, betaLocal, x1x2Local, elementCount);
            CopyOut(rowIdx, rowStep);
            gmOffset += rowStep * numLastDim;
        }
        {
            auto rowIdx = rowMoveCnt - 1;
            if (rowIdx > 0) {
                elementCount = numLastDimAligned * rowTail_;

                LocalTensor<T> x1x2LocalIn = x1x2Que.template AllocTensor<T>();
                DataCopyEx(x1x2LocalIn[0], x1Gm[gmOffset], numLastDim, rowTail_, padParams);
                DataCopyEx(x1x2LocalIn[elementCount], x2Gm[gmOffset], numLastDim, rowTail_, padParams);
                x1x2Que.EnQue(x1x2LocalIn);
                auto x1x2Local = x1x2Que.template DeQue<T>();

                if constexpr (IS_BIAS_BROADCAST) {
                    CopyInAndAddBroadCast(rowIdx, rowTail_, biasLocal, x1x2Local, elementCount);
                }
                CopyOutAdditionalOutput(rowIdx, rowTail_);
                precisionCompute(rowTail_, gammaLocal, betaLocal, x1x2Local, elementCount);
                CopyOut(rowIdx, rowTail_);
            }
        }
    }

private:
    __aicore__ inline void CopyInAndAddBroadCast(
        int32_t procId, int32_t rowCount, LocalTensor<T> &biasLocal, LocalTensor<T> &x1x2Local, uint32_t elementCount)
    {
        LocalTensor<float> addBufLocal = xBufFp32.Get<float>();
        LocalTensor<float> yBufLocal = yBufFp32.Get<float>();

        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[elementCount];

        // Use add as
        if constexpr (IsSame<float, T>::value) {
            Add(addBufLocal, x2Local, x1Local, elementCount);
            pipe_barrier(PIPE_V);
            for (int i = 0; i < rowCount; i++) {
                Add(addBufLocal[i * numLastDimAligned], biasLocal, addBufLocal[i * numLastDimAligned], numLastDim);
            }
            pipe_barrier(PIPE_V);
        } else {
            Cast(addBufLocal, x1Local, RoundMode::CAST_NONE, elementCount);
            Cast(yBufLocal, x2Local, RoundMode::CAST_NONE, elementCount);
            pipe_barrier(PIPE_V);
            Add(yBufLocal, addBufLocal, yBufLocal, elementCount);
            Cast(x1x2Local.template ReinterpretCast<float>(), biasLocal, RoundMode::CAST_NONE, numLastDim);
            pipe_barrier(PIPE_V);
            for (int i = 0; i < rowCount; i++) {
                Add(addBufLocal[i * numLastDimAligned],
                    x1x2Local.template ReinterpretCast<float>(),
                    yBufLocal[i * numLastDimAligned],
                    numLastDim);
            }
            pipe_barrier(PIPE_V);
        }
        x1x2Que.FreeTensor(x1x2Local);
    }

    __aicore__ inline void precisionCompute(int32_t nums, LocalTensor<float> &gammaLocal, LocalTensor<float> &betaLocal,
        LocalTensor<T> &x_out, uint32_t elementCount)
    {
#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> meanLocal = meanQue.template AllocTensor<float>();
        LocalTensor<float> rstdLocal = rstdQue.template AllocTensor<float>();
#endif
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        Muls(yLocalFp32, xLocalFp32, aveNum, elementCount);
        pipe_barrier(PIPE_V);

        // Reduce#1 for E(x)
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto aveLocalTemp = ReduceSumFP32(yLocalFp32[rid * numLastDimAligned], numLastDim);
#if OUTPUT_MEAN_RSTD == 1
            meanLocal.SetValue(rid, aveLocalTemp);
#endif
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            Adds(yLocalFp32[rid * numLastDimAligned],
                xLocalFp32[rid * numLastDimAligned],
                aveLocalTemp * -1,
                numLastDim);
        }
        pipe_barrier(PIPE_V);

        Mul(xLocalFp32, yLocalFp32, yLocalFp32, elementCount);
        pipe_barrier(PIPE_V);
        Muls(xLocalFp32, xLocalFp32, aveNum, elementCount);
        pipe_barrier(PIPE_V);

        // Reduce#2 for Var(x)
        for (int32_t rid = 0; rid < nums; ++rid) {
            float varLocalTemp = ReduceSumFP32(xLocalFp32[rid * numLastDimAligned], numLastDim);
            float rstdLocalTemp = 1 / sqrt(varLocalTemp + eps);
#if OUTPUT_MEAN_RSTD == 1
            rstdLocal.SetValue(rid, rstdLocalTemp);
#endif
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            Muls(xLocalFp32[rid * numLastDimAligned], yLocalFp32[rid * numLastDimAligned], rstdLocalTemp, numLastDim);
        }
        pipe_barrier(PIPE_V);
        LocalTensor<T> yLocal = yQue.template AllocTensor<T>();

        if constexpr (!IsSame<T, float>::value) {
            for (int32_t rid = 0; rid < nums; ++rid) {
                FusedMulAdd(xLocalFp32[rid * numLastDimAligned], gammaLocal, betaLocal, numLastDim);
            }
            pipe_barrier(PIPE_V);
            if constexpr (IsSame<T, half>::value) {
                Cast(yLocal, xLocalFp32, RoundMode::CAST_NONE, elementCount);
            } else {
                Cast(yLocal, xLocalFp32, RoundMode::CAST_RINT, elementCount);
            }
        } else {
            for (int32_t rid = 0; rid < nums; ++rid) {
                FusedMulAdd(xLocalFp32[rid * numLastDimAligned], gammaLocal, betaLocal, numLastDim);
            }
            pipe_barrier(PIPE_V);
            Adds(yLocal, xLocalFp32, (float)0.0, elementCount);
        }

#if OUTPUT_MEAN_RSTD == 1
        meanQue.EnQue(meanLocal);
        rstdQue.EnQue(rstdLocal);
#endif
        yQue.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t rowIdx, int32_t rowCount)
    {
        LocalTensor<T> res = yQue.template DeQue<T>();
        uint32_t gmOffset = rowIdx * rowStep * numLastDim;
        DataCopyEx(yGm[gmOffset], res, numLastDim, rowCount);
        yQue.FreeTensor(res);

#if OUTPUT_MEAN_RSTD == 1
        uint32_t gmOffsetMean = rowIdx * rowStep;
        LocalTensor<float> mean = meanQue.template DeQue<float>();
        LocalTensor<float> rstd = rstdQue.template DeQue<float>();
        DataCopyEx(meanGm[gmOffsetMean], mean, rowCount);
        DataCopyEx(rstdGm[gmOffsetMean], rstd, rowCount);
        meanQue.FreeTensor(mean);
        rstdQue.FreeTensor(rstd);
#endif
    }

    __aicore__ inline void CopyOutAdditionalOutput(int32_t procId, int32_t rowCount)
    {
        if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
            LocalTensor<float> addBufLocal = xBufFp32.Get<float>();
            uint32_t gmOffset = procId * rowStep * numLastDim;
            auto elementCount = numLastDimAligned * rowCount;
            auto xLocal = yQue.template AllocTensor<T>();
            if constexpr (IsSame<T, float>::value) {
                Adds(xLocal, addBufLocal, ZERO, elementCount);
            } else if constexpr (IsSame<T, half>::value) {
                Cast(xLocal, addBufLocal, RoundMode::CAST_NONE, elementCount);
            } else {
                Cast(xLocal, addBufLocal, RoundMode::CAST_RINT, elementCount);
            }
            pipe_barrier(PIPE_V);
            yQue.template EnQue<T>(xLocal);
            auto x = yQue.template DeQue<T>();

            DataCopyEx(xGm[gmOffset], x, numLastDim, rowCount);
            yQue.FreeTensor(x);
        }
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1x2Que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue;
#if OUTPUT_MEAN_RSTD == 1
    TQue<QuePosition::VECOUT, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstdQue;
#endif
    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;

    TBuf<TPosition::VECCALC> gammaBuf;
    TBuf<TPosition::VECCALC> betaBuf;
    TBuf<TPosition::VECCALC> biasBuf;

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> rstdGm;
    uint32_t numCore;
    uint32_t numFirstDim;
    uint32_t numLastDim;
    uint32_t rowStep;
    uint32_t rowWork;
    uint32_t gmOffset_;
    uint32_t rowTail_;
    uint32_t colTail;
    uint32_t colMoveCnt;
    uint32_t firstDimPerTime;
    uint32_t lastDimPerTime;
    uint32_t nlFirstDimPerCore;
    uint32_t lFirstDimPerCore;
    float eps;
    float aveNum;
    bool lastDimPad = false;
    size_t numLastDimAligned;
};

template <typename T_X1, typename T_X2, typename T_GAMMA, typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormNormalSpecialReduce {
#define IS_ADDITIONAL_OUTPUT_ENABLE ((TILING_KEY % 1000) / 100 == 1)
#define IS_NORMAL_SPECIAL_REDUCE_BIG_N_CASE ((TILING_KEY % 100) / 10 == 8)
#define IS_BIAS_BROADCAST ((TILING_KEY % 10) == 2)

public:
    __aicore__ inline KernelAddLayerNormNormalSpecialReduce(TPipe *pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y > 0) {
            return (x + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline uint32_t ROUND_UP32(uint32_t x)
    {
        return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }

    __aicore__ inline uint32_t BlockAlign(uint32_t x, uint32_t blockElem)
    {
        if (blockElem > 0) {
            return (x + blockElem - 1) / blockElem * blockElem;
        }
        return 0;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return x < y ? x : y;
    }

    __aicore__ inline uint32_t MAX(uint32_t x, uint32_t y)
    {
        return x > y ? x : y;
    }

    __aicore__ inline void Init(__gm__ uint8_t *x1, __gm__ uint8_t *x2, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
        __gm__ uint8_t *bias, __gm__ uint8_t *y, __gm__ uint8_t *mean, __gm__ uint8_t *rstd, __gm__ uint8_t *x,
        __gm__ uint8_t *workspace, uint32_t numCore_, uint32_t numLastDim_, uint32_t numFirstDim_,
        uint32_t nlFirstDimPerCore_, uint32_t lFirstDimPerCore_, uint32_t firstDimPerTime_, uint32_t lastDimPerTime_,
        float eps_, float aveNum_, uint32_t colMoveCnt_, uint32_t colTail_, uint32_t workspace_size)
    {
        numCore = numCore_;
        numLastDim = numLastDim_;
        numFirstDim = numFirstDim_;
        notLastFirstDimPerCore = nlFirstDimPerCore_;
        lFirstDimPerCore = lFirstDimPerCore_;
        firstDimPerTime = firstDimPerTime_;
        lastDimPerTime = lastDimPerTime_;
        aveNum = aveNum_;
        eps = eps_;
        colMoveCnt = colMoveCnt_;
        colTail = colTail_;
        if (block_idx != numCore - 1) {
            rowWork = notLastFirstDimPerCore;
            rowStep = firstDimPerTime;
        } else {
            rowWork = lFirstDimPerCore;
            rowStep = MIN(firstDimPerTime, rowWork);
        }
        rowTail_ = (rowWork % rowStep == 0) ? rowStep : (rowWork % rowStep);
        gmOffset_ = notLastFirstDimPerCore * numLastDim;
        x1Gm.SetGlobalBuffer((__gm__ T *)(x1) + block_idx * gmOffset_);
        x2Gm.SetGlobalBuffer((__gm__ T *)(x2) + block_idx * gmOffset_);
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma);
        betaGm.SetGlobalBuffer((__gm__ T *)beta);
        yGm.SetGlobalBuffer((__gm__ T *)(y) + block_idx * gmOffset_);
        // mean/rstd always output fp32
        meanGm.SetGlobalBuffer((__gm__ float *)mean + block_idx * notLastFirstDimPerCore);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + block_idx * notLastFirstDimPerCore);
        xGm.SetGlobalBuffer((__gm__ T *)(x) + block_idx * gmOffset_);
        if constexpr (IS_BIAS_BROADCAST) {
            biasGm.SetGlobalBuffer((__gm__ T *)bias);
        }

        numLastDimAligned = numLastDim;
        if (ROUND_UP32(numLastDim * sizeof(T)) != numLastDim * sizeof(T)) {
            lastDimPad = true;
            numLastDimAligned = ROUND_UP32(numLastDim * sizeof(T)) / sizeof(T);
        }

        Ppipe->InitBuffer(x1x2Que, BUFFER_NUM, ROUND_UP32(2 * rowStep * numLastDimAligned * sizeof(T)));
        Ppipe->InitBuffer(yQue, BUFFER_NUM, ROUND_UP32(rowStep * numLastDimAligned * sizeof(T)));
        Ppipe->InitBuffer(betaBuf, ROUND_UP32(numLastDimAligned * sizeof(float)));
        Ppipe->InitBuffer(gammaBuf, ROUND_UP32(numLastDimAligned * sizeof(float)));
        if constexpr (IS_BIAS_BROADCAST) {
            Ppipe->InitBuffer(biasBuf, ROUND_UP32(numLastDim * sizeof(T)));
        }

        Ppipe->InitBuffer(xBufFp32, ROUND_UP32(rowStep * numLastDimAligned * sizeof(float)));
        Ppipe->InitBuffer(zBufFp32, ROUND_UP32(rowStep * numLastDimAligned * sizeof(float)));
#if __CCE_AICORE__ == 220
        uint32_t brcbRowStep = BlockAlign(rowStep, BRCB_BROADCAST_NUMBER);
        Ppipe->InitBuffer(yBufFp32, ROUND_UP32(brcbRowStep * ELEM_PER_REP_FP32 * sizeof(float)));
#else
        Ppipe->InitBuffer(yBufFp32, ROUND_UP32(rowStep * ELEM_PER_REP_FP32 * sizeof(float)));
#endif

#if __CCE_AICORE__ != 220
        Ppipe->InitBuffer(orBufINT16, 16 * sizeof(int16_t));  // one block

        if constexpr (IS_NORMAL_SPECIAL_REDUCE_BIG_N_CASE) {
            Ppipe->InitBuffer(transposeSrcBuf, ROUND_UP32(16 * 16 * 8 * sizeof(half)));
            Ppipe->InitBuffer(transposeDstBuf, ROUND_UP32(16 * 16 * 8 * sizeof(half)));
        }
#endif

#if OUTPUT_MEAN_RSTD == 1
        Ppipe->InitBuffer(meanQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
        Ppipe->InitBuffer(rstdQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
#endif
    }

    __aicore__ inline void Process()
    {
        int32_t rowMoveCnt = CEIL_DIV(rowWork, rowStep);

        DataCopyPadParams padParams;
        if (lastDimPad) {
            padParams.isPad = true;
            padParams.paddingValue = 0;
            padParams.rightPadding = numLastDimAligned - numLastDim;
        }

        LocalTensor<float> betaLocal = betaBuf.template Get<float>();
        LocalTensor<float> gammaLocal = gammaBuf.template Get<float>();

        if constexpr (IsSame<float, T>::value) {
            DataCopyEx(betaLocal, betaGm, numLastDim);
            DataCopyEx(gammaLocal, gammaGm, numLastDim);
        } else {
            auto betaLocalHalf = betaLocal.ReinterpretCast<T>();
            auto gammaLocalHalf = gammaLocal.ReinterpretCast<T>();
            DataCopyEx(betaLocalHalf[numLastDimAligned], betaGm, numLastDim);
            DataCopyEx(gammaLocalHalf[numLastDimAligned], gammaGm, numLastDim);
        }

        LocalTensor<T> biasLocal;
        if constexpr (IS_BIAS_BROADCAST) {
            biasLocal = biasBuf.template Get<T>();
            DataCopyEx(biasLocal, biasGm, numLastDim);
        }

        uint32_t gmOffset = 0;
        uint32_t elementCount = numLastDimAligned * rowStep;

        {
            LocalTensor<T> x1x2LocalIn = x1x2Que.template AllocTensor<T>();
            DataCopyEx(x1x2LocalIn[0], x1Gm[gmOffset], numLastDim, rowStep, padParams);
            DataCopyEx(x1x2LocalIn[elementCount], x2Gm[gmOffset], numLastDim, rowStep, padParams);
            x1x2Que.EnQue(x1x2LocalIn);
            auto x1x2Local = x1x2Que.template DeQue<T>();

            if constexpr (!IsSame<T, float>::value) {
                Cast(gammaLocal, gammaLocal.ReinterpretCast<T>()[numLastDimAligned], RoundMode::CAST_NONE, numLastDim);
                Cast(betaLocal, betaLocal.ReinterpretCast<T>()[numLastDimAligned], RoundMode::CAST_NONE, numLastDim);
            }

            if constexpr (IS_BIAS_BROADCAST) {
                CopyInAndAddBroadCast(rowStep, elementCount, biasLocal, x1x2Local);
            } else {
                CopyIn(rowStep, elementCount, x1x2Local, padParams);
            }
            CopyOutAdditionalOutput(0, rowStep);
            if constexpr (IS_NORMAL_SPECIAL_REDUCE_BIG_N_CASE) {
                PrecisionComputeBigN(rowStep, gammaLocal, betaLocal);
            } else {
                PrecisionCompute(rowStep, gammaLocal, betaLocal, elementCount);
            }
            CopyOut(0, rowStep);
            gmOffset += rowStep * numLastDim;
        }
        for (int32_t rowIdx = 1; rowIdx < rowMoveCnt - 1; ++rowIdx) {
            LocalTensor<T> x1x2LocalIn = x1x2Que.template AllocTensor<T>();
            DataCopyEx(x1x2LocalIn[0], x1Gm[gmOffset], numLastDim, rowStep, padParams);
            DataCopyEx(x1x2LocalIn[elementCount], x2Gm[gmOffset], numLastDim, rowStep, padParams);
            x1x2Que.EnQue(x1x2LocalIn);
            auto x1x2Local = x1x2Que.template DeQue<T>();
            if constexpr (IS_BIAS_BROADCAST) {
                CopyInAndAddBroadCast(rowStep, elementCount, biasLocal, x1x2Local);
            } else {
                CopyIn(rowStep, elementCount, x1x2Local, padParams);
            }

            CopyOutAdditionalOutput(rowIdx, rowStep);
            if constexpr (IS_NORMAL_SPECIAL_REDUCE_BIG_N_CASE) {
                PrecisionComputeBigN(rowStep, gammaLocal, betaLocal);
            } else {
                PrecisionCompute(rowStep, gammaLocal, betaLocal, elementCount);
            }
            CopyOut(rowIdx, rowStep);
            gmOffset += rowStep * numLastDim;
        }
        {
            auto rowIdx = rowMoveCnt - 1;
            if (rowIdx > 0) {
                elementCount = numLastDimAligned * rowTail_;

                LocalTensor<T> x1x2LocalIn = x1x2Que.template AllocTensor<T>();
                DataCopyEx(x1x2LocalIn[0], x1Gm[gmOffset], numLastDim, rowTail_, padParams);
                DataCopyEx(x1x2LocalIn[elementCount], x2Gm[gmOffset], numLastDim, rowTail_, padParams);
                x1x2Que.EnQue(x1x2LocalIn);
                auto x1x2Local = x1x2Que.template DeQue<T>();
                if constexpr (IS_BIAS_BROADCAST) {
                    CopyInAndAddBroadCast(rowTail_, elementCount, biasLocal, x1x2Local);
                } else {
                    CopyIn(rowTail_, elementCount, x1x2Local, padParams);
                }

                CopyOutAdditionalOutput(rowIdx, rowTail_);
                if constexpr (IS_NORMAL_SPECIAL_REDUCE_BIG_N_CASE) {
                    PrecisionComputeBigN(rowTail_, gammaLocal, betaLocal);
                } else {
                    PrecisionCompute(rowTail_, gammaLocal, betaLocal, elementCount);
                }
                CopyOut(rowIdx, rowTail_);
            }
        }
    }

private:
    __aicore__ inline void CopyInAndAddBroadCast(
        int32_t rowCount, uint32_t elementCount, LocalTensor<T> &biasLocal, LocalTensor<T> &x1x2Local)
    {
        LocalTensor<float> addBufLocal = zBufFp32.Get<float>();
        LocalTensor<float> xBufLocal = xBufFp32.Get<float>();

        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[elementCount];

        // Use add as
        if constexpr (IsSame<float, T>::value) {
            Add(addBufLocal, x2Local, x1Local, elementCount);
            pipe_barrier(PIPE_V);
            for (int i = 0; i < rowCount; i++) {
                Add(addBufLocal[i * numLastDimAligned], biasLocal, addBufLocal[i * numLastDimAligned], numLastDim);
            }
            pipe_barrier(PIPE_V);
        } else {
            Cast(addBufLocal, x1Local, RoundMode::CAST_NONE, elementCount);
            Cast(xBufLocal, x2Local, RoundMode::CAST_NONE, elementCount);
            pipe_barrier(PIPE_V);
            Add(xBufLocal, addBufLocal, xBufLocal, elementCount);
            Cast(x1x2Local.template ReinterpretCast<float>(), biasLocal, RoundMode::CAST_NONE, numLastDim);
            pipe_barrier(PIPE_V);
            for (int i = 0; i < rowCount; i++) {
                Add(addBufLocal[i * numLastDimAligned],
                    x1x2Local.template ReinterpretCast<float>(),
                    xBufLocal[i * numLastDimAligned],
                    numLastDim);
            }
            pipe_barrier(PIPE_V);
        }
        x1x2Que.FreeTensor(x1x2Local);
    }

    __aicore__ inline void CopyIn(
        int32_t rowCount, uint32_t elementCount, LocalTensor<T> &x1x2Local, const DataCopyPadParams &padParams = {})
    {
        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[elementCount];

        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> addBufLocal = zBufFp32.Get<float>();

        // Use add as
        if constexpr (IsSame<float, T>::value) {
            Add(addBufLocal, x2Local, x1Local, elementCount);
            pipe_barrier(PIPE_V);
        } else {
            Cast(addBufLocal, x1Local, RoundMode::CAST_NONE, elementCount);
            Cast(xLocalFp32, x2Local, RoundMode::CAST_NONE, elementCount);
            pipe_barrier(PIPE_V);
            Add(addBufLocal, addBufLocal, xLocalFp32, elementCount);
            pipe_barrier(PIPE_V);
        }
        x1x2Que.FreeTensor(x1x2Local);
    }

    __aicore__ inline void CopyOutAdditionalOutput(int32_t procId, int32_t rowCount)
    {
        if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
            LocalTensor<float> addBufLocal = zBufFp32.Get<float>();
            uint32_t gmOffset = procId * rowStep * numLastDim;
            auto elementCount = numLastDimAligned * rowCount;
            auto xLocal = yQue.template AllocTensor<T>();
            if constexpr (IsSame<T, float>::value) {
                Adds(xLocal, addBufLocal, ZERO, elementCount);
            } else if constexpr (IsSame<T, half>::value) {
                Cast(xLocal, addBufLocal, RoundMode::CAST_NONE, elementCount);
            } else {
                Cast(xLocal, addBufLocal, RoundMode::CAST_RINT, elementCount);
            }
            pipe_barrier(PIPE_V);
            yQue.template EnQue<T>(xLocal);

            auto x = yQue.template DeQue<T>();
            DataCopyEx(xGm[gmOffset], x, numLastDim, rowCount);
            yQue.FreeTensor(x);
        }
    }

    __aicore__ inline void PrecisionCompute(
        int32_t nums, LocalTensor<float> &gammaLocal, LocalTensor<float> &betaLocal, uint32_t elementCount)
    {
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> meanLocal = meanQue.template AllocTensor<float>();
        LocalTensor<float> rstdLocal = rstdQue.template AllocTensor<float>();
#endif
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();  // for reduce
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();

        // 1.1. mean process: 1/N * x_sum
        Muls(xLocalFp32, zLocalFp32, aveNum, elementCount);
        // 1.2. init buffer for reduce
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        pipe_barrier(PIPE_V);

        // 2. mean end: reduce(1/N * x_sum)
        const uint32_t tailCount = numLastDim % ELEM_PER_REP_FP32;
        const uint32_t repeat = nums;  // repeat < 255 * 8 = 2040
        const uint8_t repStride = numLastDimAligned / FLOAT_BLOCK_ELEM;
        ReduceSumForSmallReduceDim(
            xLocalFp32, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);

        // 3. rstd process: x - mean
        set_flag(PIPE_V, PIPE_S, eventVS);
        wait_flag(PIPE_V, PIPE_S, eventVS);
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * numLastDimAligned;

            auto meanTemp = xLocalFp32.GetValue(rid);
#if OUTPUT_MEAN_RSTD == 1
            meanLocal.SetValue(rid, meanTemp);
#endif
            set_flag(PIPE_S, PIPE_V, eventSV);
            wait_flag(PIPE_S, PIPE_V, eventSV);
            Adds(zLocalFp32[roundOffset], zLocalFp32[roundOffset], meanTemp * -1, numLastDim);
        }
        pipe_barrier(PIPE_V);

        // 4. rstd process: (x - mean) ^ 2
        Mul(xLocalFp32, zLocalFp32, zLocalFp32, elementCount);
        pipe_barrier(PIPE_V);

        // 5. rstd process: reduce( 1 / N * (x - mean) ^ 2 )
        Muls(xLocalFp32, xLocalFp32, aveNum, elementCount);
        pipe_barrier(PIPE_V);
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        pipe_barrier(PIPE_V);
        ReduceSumForSmallReduceDim(
            xLocalFp32, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);
        pipe_barrier(PIPE_V);

        // 6. rstd end: 1 / sqrt( 1 / N * reduce( (x - mean) ^ 2 ) )
        Adds(xLocalFp32, xLocalFp32, eps, repeat);
        pipe_barrier(PIPE_V);
        Sqrt(xLocalFp32, xLocalFp32, repeat);
        Duplicate(yLocalFp32, float(1), repeat);
        pipe_barrier(PIPE_V);
        Div(xLocalFp32, yLocalFp32, xLocalFp32, repeat);

        // 7. y process: (x - mean) / rstd
        set_flag(PIPE_V, PIPE_S, eventVS);
        wait_flag(PIPE_V, PIPE_S, eventVS);
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * numLastDimAligned;

            float rstdTmp = xLocalFp32.GetValue(rid);
#if OUTPUT_MEAN_RSTD == 1
            rstdLocal.SetValue(rid, rstdTmp);
#endif
            set_flag(PIPE_S, PIPE_V, eventSV);
            wait_flag(PIPE_S, PIPE_V, eventSV);
            Muls(zLocalFp32[roundOffset], zLocalFp32[roundOffset], rstdTmp, numLastDim);
        }
        pipe_barrier(PIPE_V);

        // 8. y = (x - mean) / rstd * beta + gamma
        LocalTensor<T> yLocal = yQue.template AllocTensor<T>();
        if constexpr (!IsSame<T, float>::value) {
            for (int32_t rid = 0; rid < nums; ++rid) {
                Mul(zLocalFp32[rid * numLastDimAligned], zLocalFp32[rid * numLastDimAligned], gammaLocal, numLastDim);
            }
            pipe_barrier(PIPE_V);
            for (int32_t rid = 0; rid < nums; ++rid) {
                Add(zLocalFp32[rid * numLastDimAligned], zLocalFp32[rid * numLastDimAligned], betaLocal, numLastDim);
            }
            pipe_barrier(PIPE_V);

            if constexpr (IsSame<T, half>::value) {
                Cast(yLocal, zLocalFp32, RoundMode::CAST_NONE, elementCount);
            } else {
                Cast(yLocal, zLocalFp32, RoundMode::CAST_RINT, elementCount);
            }
        } else {
            for (int32_t rid = 0; rid < nums; ++rid) {
                Mul(zLocalFp32[rid * numLastDimAligned], zLocalFp32[rid * numLastDimAligned], gammaLocal, numLastDim);
            }
            pipe_barrier(PIPE_V);
            for (int32_t rid = 0; rid < nums; ++rid) {
                Add(yLocal[rid * numLastDimAligned], zLocalFp32[rid * numLastDimAligned], betaLocal, numLastDim);
            }
            pipe_barrier(PIPE_V);
        }

#if OUTPUT_MEAN_RSTD == 1
        meanQue.EnQue(meanLocal);
        rstdQue.EnQue(rstdLocal);
#endif
        yQue.EnQue(yLocal);
    }

    __aicore__ inline void PrecisionComputeBigN(
        int32_t nums, LocalTensor<float> &gammaLocal, LocalTensor<float> &betaLocal)
    {
#if __CCE_AICORE__ == 220
        PrecisionComputeBigNBrcb(nums, gammaLocal, betaLocal);
#else
        precisionComputeBigNTranspose(nums, gammaLocal, betaLocal);
#endif
    }

    __aicore__ inline void precisionComputeBigNTranspose(
        int32_t nums, LocalTensor<float> &gammaLocal, LocalTensor<float> &betaLocal)
    {
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

        LocalTensor<int16_t> orOffsetINT16 = orBufINT16.Get<int16_t>();

        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();

        LocalTensor<float> transposeSrcLocal = transposeSrcBuf.Get<float>();
        LocalTensor<float> transposeDstLocal = transposeDstBuf.Get<float>();

        int32_t elementNum = numLastDimAligned * nums;

        // 1.1. mean process: 1/N * x_sum
        Muls(xLocalFp32, zLocalFp32, aveNum, elementNum);
        // 1.2. init buffer for reduce
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        pipe_barrier(PIPE_V);

        // 2.1. reducesum
        const uint32_t forCount = numLastDim / ELEM_PER_REP_FP32;
        const uint32_t tailCount = numLastDim % ELEM_PER_REP_FP32;
        const uint32_t repeat = nums;  // repeat < 255 * 8 = 2040
        const uint8_t repStride = numLastDimAligned / FLOAT_BLOCK_ELEM;
        ReduceSumForSmallReduceDim(
            xLocalFp32, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);
        pipe_barrier(PIPE_V);

        // 2.2. broadcast reducesum value
        InitVAForTranspose(
            (__ubuf__ half *)transposeDstLocal.GetPhyAddr(), (__ubuf__ half *)transposeSrcLocal.GetPhyAddr());
        CCEBroadCastShort((__ubuf__ int16_t *)xLocalFp32.GetPhyAddr(),
            (__ubuf__ float *)xLocalFp32.GetPhyAddr(),
            (__ubuf__ int16_t *)transposeDstLocal.GetPhyAddr(),
            (__ubuf__ int16_t *)transposeSrcLocal.GetPhyAddr(),
            (__ubuf__ int16_t *)orOffsetINT16.GetPhyAddr(),
            forCount,
            tailCount,
            repeat,
            repStride);
        pipe_barrier(PIPE_V);

        // 3. rstd process: x - mean
        Sub(zLocalFp32, zLocalFp32, xLocalFp32, elementNum);
        pipe_barrier(PIPE_V);

        // 4. rstd process: (x - mean) ^ 2
        Mul(xLocalFp32, zLocalFp32, zLocalFp32, elementNum);
        pipe_barrier(PIPE_V);

        // 5. rstd process: reduce( 1 / N * (x - mean) ^ 2 )
        Muls(xLocalFp32, xLocalFp32, aveNum, elementNum);
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        pipe_barrier(PIPE_V);
        ReduceSumForSmallReduceDim(
            xLocalFp32, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);
        pipe_barrier(PIPE_V);

        // 6. rstd end: 1 / sqrt( 1 / N * reduce( (x - mean) ^ 2 ) )
        Adds(xLocalFp32, xLocalFp32, eps, nums);
        pipe_barrier(PIPE_V);
        Sqrt(xLocalFp32, xLocalFp32, nums);
        Duplicate(yLocalFp32, (float)1, nums);
        pipe_barrier(PIPE_V);
        Div(xLocalFp32, yLocalFp32, xLocalFp32, nums);
        pipe_barrier(PIPE_V);

        // 7. broadcast reducesum value
        CCEBroadCastShort((__ubuf__ int16_t *)xLocalFp32.GetPhyAddr(),
            (__ubuf__ float *)xLocalFp32.GetPhyAddr(),
            (__ubuf__ int16_t *)transposeDstLocal.GetPhyAddr(),
            (__ubuf__ int16_t *)transposeSrcLocal.GetPhyAddr(),
            (__ubuf__ int16_t *)orOffsetINT16.GetPhyAddr(),
            forCount,
            tailCount,
            repeat,
            repStride);
        pipe_barrier(PIPE_V);
        Mul(zLocalFp32, zLocalFp32, xLocalFp32, elementNum);
        pipe_barrier(PIPE_V);

        // 8. y = (x - mean) / rstd * beta + gamma
        LocalTensor<T> yLocal = yQue.template AllocTensor<T>();
        if constexpr (!IsSame<T, float>::value) {
            Level0MulFp32Short(zLocalFp32, gammaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            pipe_barrier(PIPE_V);
            Level0AddFp32Short(zLocalFp32, betaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            pipe_barrier(PIPE_V);

            if constexpr (IsSame<T, half>::value) {
                Cast(yLocal, zLocalFp32, RoundMode::CAST_NONE, elementNum);
            } else {
                Cast(yLocal, zLocalFp32, RoundMode::CAST_RINT, elementNum);
            }
            pipe_barrier(PIPE_V);
        } else {
            Level0MulFp32Short(yLocal, gammaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            pipe_barrier(PIPE_V);
            Level0AddFp32Short(yLocal, betaLocal, yLocal, numLastDimAligned, nums, numLastDim);
            pipe_barrier(PIPE_V);
        }

        yQue.EnQue(yLocal);
    }

#if __CCE_AICORE__ == 220
    __aicore__ inline void PrecisionComputeBigNBrcb(
        int32_t nums, LocalTensor<float> &gammaLocal, LocalTensor<float> &betaLocal)
    {
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> zLocalFp32 = zBufFp32.Get<float>();

        LocalTensor<float> meanLocal = meanQue.template AllocTensor<float>();
        LocalTensor<float> rstdLocal = rstdQue.template AllocTensor<float>();

        int32_t elementNum = numLastDimAligned * nums;

        // 1.1. mean process: 1/N * x_sum
        Muls(xLocalFp32, zLocalFp32, aveNum, elementNum);
        // 1.2. init buffer for reduce
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        pipe_barrier(PIPE_V);

        // 2.1. reducesum
        const uint32_t tailCount = numLastDim % ELEM_PER_REP_FP32;
        const uint32_t repeat = nums;  // repeat < 255 * 8 = 2040
        const uint8_t repStride = numLastDimAligned / FLOAT_BLOCK_ELEM;
        ReduceSumForSmallReduceDim(
            meanLocal, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);
        pipe_barrier(PIPE_V);

        // 2.2. broadcast reducesum value
        const uint32_t broadcastDim = BROADCAST_ND_DIM_NUM;
        const uint32_t broadcastAxis = BROADCAST_ND_LAST_INDEX;
        uint32_t dstShape[broadcastDim] = {(uint32_t)nums, (uint32_t)numLastDimAligned};
        uint32_t srcShape[broadcastDim] = {(uint32_t)nums, 1};
        auto sharedTmpBuffer = yLocalFp32.ReinterpretCast<uint8_t>();
        BroadCast<float, broadcastDim, broadcastAxis>(xLocalFp32, meanLocal, dstShape, srcShape, sharedTmpBuffer);
        pipe_barrier(PIPE_V);

        // 3. rstd process: x - mean
        Sub(zLocalFp32, zLocalFp32, xLocalFp32, elementNum);
        pipe_barrier(PIPE_V);

        // 4. rstd process: (x - mean) ^ 2
        Mul(xLocalFp32, zLocalFp32, zLocalFp32, elementNum);
        pipe_barrier(PIPE_V);

        // 5. rstd process: reduce( 1 / N * (x - mean) ^ 2 )
        Muls(xLocalFp32, xLocalFp32, aveNum, elementNum);
        Duplicate(yLocalFp32, ZERO, nums * ELEM_PER_REP_FP32);
        pipe_barrier(PIPE_V);
        ReduceSumForSmallReduceDim(
            rstdLocal, xLocalFp32, yLocalFp32, numLastDimAligned, numLastDim, tailCount, repeat, repStride);
        pipe_barrier(PIPE_V);

        // 6. rstd end: 1 / sqrt( 1 / N * reduce( (x - mean) ^ 2 ) )
        Adds(rstdLocal, rstdLocal, eps, nums);
        pipe_barrier(PIPE_V);
        Sqrt(rstdLocal, rstdLocal, nums);
        Duplicate(yLocalFp32, (float)1, nums);
        pipe_barrier(PIPE_V);
        Div(rstdLocal, yLocalFp32, rstdLocal, nums);
        pipe_barrier(PIPE_V);

        // 7. broadcast reducesum value
        BroadCast<float, broadcastDim, broadcastAxis>(xLocalFp32, rstdLocal, dstShape, srcShape, sharedTmpBuffer);
        pipe_barrier(PIPE_V);
        Mul(zLocalFp32, zLocalFp32, xLocalFp32, elementNum);
        pipe_barrier(PIPE_V);

        // 8. y = (x - mean) / rstd * beta + gamma
        LocalTensor<T> yLocal = yQue.template AllocTensor<T>();
        if constexpr (!IsSame<T, float>::value) {
            Level0MulFp32Short(zLocalFp32, gammaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            pipe_barrier(PIPE_V);
            Level0AddFp32Short(zLocalFp32, betaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            pipe_barrier(PIPE_V);

            if constexpr (IsSame<T, half>::value) {
                Cast(yLocal, zLocalFp32, RoundMode::CAST_NONE, elementNum);
            } else {
                Cast(yLocal, zLocalFp32, RoundMode::CAST_RINT, elementNum);
            }
            pipe_barrier(PIPE_V);
        } else {
            Level0MulFp32Short(yLocal, gammaLocal, zLocalFp32, numLastDimAligned, nums, numLastDim);
            pipe_barrier(PIPE_V);
            Level0AddFp32Short(yLocal, betaLocal, yLocal, numLastDimAligned, nums, numLastDim);
            pipe_barrier(PIPE_V);
        }

        meanQue.EnQue(meanLocal);
        rstdQue.EnQue(rstdLocal);
        yQue.EnQue(yLocal);
    }
#endif

    __aicore__ inline void CopyOut(int32_t rowIdx, int32_t rowCount)
    {
        LocalTensor<T> res = yQue.template DeQue<T>();
        uint32_t gmOffset = rowIdx * rowStep * numLastDim;
        DataCopyEx(yGm[gmOffset], res, numLastDim, rowCount);
        yQue.FreeTensor(res);

#if OUTPUT_MEAN_RSTD == 1
        uint32_t gmOffsetMean = rowIdx * rowStep;
        LocalTensor<float> mean = meanQue.template DeQue<float>();
        LocalTensor<float> rstd = rstdQue.template DeQue<float>();
        DataCopyEx(meanGm[gmOffsetMean], mean, rowCount);
        DataCopyEx(rstdGm[gmOffsetMean], rstd, rowCount);
        meanQue.FreeTensor(mean);
        rstdQue.FreeTensor(rstd);
#endif
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1x2Que;
    TBuf<TPosition::VECCALC> gammaBuf;
    TBuf<TPosition::VECCALC> betaBuf;
    TBuf<TPosition::VECCALC> biasBuf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue;  // (x1 + x2) reuse this que
#if OUTPUT_MEAN_RSTD == 1
    TQue<QuePosition::VECOUT, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstdQue;
#endif

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;
    TBuf<TPosition::VECCALC> zBufFp32;

    TBuf<TPosition::VECCALC> orBufINT16;
    TBuf<TPosition::VECCALC> transposeSrcBuf;
    TBuf<TPosition::VECCALC> transposeDstBuf;

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> workspaceGm;
    uint32_t numCore;
    uint32_t numFirstDim;
    uint32_t numLastDim;
    uint32_t rowStep;
    uint32_t rowWork;
    uint32_t gmOffset_;
    uint32_t rowTail_;
    uint32_t colTail;
    uint32_t colMoveCnt;
    uint32_t firstDimPerTime;
    uint32_t lastDimPerTime;
    uint32_t notLastFirstDimPerCore;
    uint32_t lFirstDimPerCore;
    float eps;
    float aveNum;
    bool lastDimPad = false;
    size_t numLastDimAligned;
    size_t numLastDimAlignedFp32;
};

template <typename T_X1, typename T_X2, typename T_GAMMA, typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormSingleRowLessTensor {
#define IS_ADDITIONAL_OUTPUT_ENABLE ((TILING_KEY % 1000) / 100 == 1)
#define IS_NO_BIAS ((TILING_KEY % 10) == 0)
#define IS_BIAS_PRESENT ((TILING_KEY % 10) == 1)
#define IS_BIAS_BROADCAST ((TILING_KEY % 10) == 2)
#define IS_SINGLE_ROW_LESS_TENSOR_CASE ((TILING_KEY % 100) / 10 == 9)
#define IS_CAST_BEFORE_ADD (!IsSame<T_X1, T_X2>::value)
#define IS_X1_NEEDCAST ((!IsSame<T_X1, float>::value) && IS_CAST_BEFORE_ADD)
#define IS_X2_NEEDCAST ((!IsSame<T_X2, float>::value) && IS_CAST_BEFORE_ADD)
#define IS_GAMMA_FP32 (IsSame<T_GAMMA, float>::value)
#define IS_X1_X2_ALL_FP32 ((IsSame<T_X1, float>::value) && (IsSame<T_X2, float>::value))
#define IS_X_B16_GAMMA_B32 ((!IS_CAST_BEFORE_ADD) && (!IsSame<T_X1, float>::value) && (IS_GAMMA_FP32))

public:
    __aicore__ inline KernelAddLayerNormSingleRowLessTensor(TPipe *pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y > 0) {
            return (x + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline uint32_t ROUND_UP32(uint32_t x)
    {
        return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return x < y ? x : y;
    }

    __aicore__ inline void Init(__gm__ uint8_t *x1, __gm__ uint8_t *x2, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
        __gm__ uint8_t *bias, __gm__ uint8_t *y, __gm__ uint8_t *mean, __gm__ uint8_t *rstd, __gm__ uint8_t *x,
        __gm__ uint8_t *workspace, uint32_t numCore_, uint32_t numLastDim_, uint32_t numFirstDim_,
        uint32_t nlFirstDimPerCore_, uint32_t lFirstDimPerCore_, uint32_t firstDimPerTime_, uint32_t lastDimPerTime_,
        float eps_, float aveNum_, uint32_t colMoveCnt_, uint32_t colTail_, uint32_t workspace_size)
    {
        numCore = numCore_;
        numLastDim = numLastDim_;
        numFirstDim = numFirstDim_;
        nlFirstDimPerCore = nlFirstDimPerCore_;
        lFirstDimPerCore = lFirstDimPerCore_;
        firstDimPerTime = firstDimPerTime_;
        lastDimPerTime = lastDimPerTime_;
        aveNum = aveNum_;
        eps = eps_;
        colMoveCnt = colMoveCnt_;
        colTail = colTail_;
        if (block_idx != numCore - 1) {
            rowWork = nlFirstDimPerCore;
            rowStep = firstDimPerTime;
        } else {
            rowWork = lFirstDimPerCore;
            rowStep = MIN(firstDimPerTime, rowWork);
        }
        rowTail_ = (rowWork % rowStep == 0) ? rowStep : (rowWork % rowStep);

        InitInputGMBuffer(x1, x2, gamma, beta, bias);
        InitOutputGMBuffer(y, mean, rstd, x);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace + workspace_size);

        numLastDimAligned = numLastDim;
        if (ROUND_UP32(numLastDim * sizeof(T)) != numLastDim * sizeof(T)) {
            lastDimPad = true;
            numLastDimAligned = ROUND_UP32(numLastDim * sizeof(T)) / sizeof(T);
        }
        if constexpr (IS_X1_NEEDCAST || IS_X2_NEEDCAST) {
            numLastDimAlignedMixDtype = numLastDim;
            if (ROUND_UP32(numLastDim * sizeof(half)) != numLastDim * sizeof(half)) {
                lastDimPadMixDtype = true;
                numLastDimAlignedMixDtype = ROUND_UP32(numLastDim * sizeof(half)) / sizeof(half);
            }
        }

        InitUBBuffer();
    }

    __aicore__ inline void InitInputGMBuffer(
        __gm__ uint8_t *x1, __gm__ uint8_t *x2, __gm__ uint8_t *gamma, __gm__ uint8_t *beta, __gm__ uint8_t *bias)
    {
        uint32_t gmOffset_ = nlFirstDimPerCore * numLastDim;
        x1Gm.SetGlobalBuffer((__gm__ T_X1 *)(x1) + block_idx * gmOffset_);
        x2Gm.SetGlobalBuffer((__gm__ T_X2 *)(x2) + block_idx * gmOffset_);
        if constexpr (IS_BIAS_PRESENT) {
            biasGm.SetGlobalBuffer((__gm__ T *)(bias) + block_idx * gmOffset_);
        } else if constexpr (IS_BIAS_BROADCAST) {
            biasGm.SetGlobalBuffer((__gm__ T *)bias);
        }
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA *)gamma);
        betaGm.SetGlobalBuffer((__gm__ T_GAMMA *)beta);
    }

    __aicore__ inline void InitOutputGMBuffer(
        __gm__ uint8_t *y, __gm__ uint8_t *mean, __gm__ uint8_t *rstd, __gm__ uint8_t *x)
    {
        uint32_t gmOffset_ = nlFirstDimPerCore * numLastDim;
        yGm.SetGlobalBuffer((__gm__ T *)(y) + block_idx * gmOffset_);
        // mean/rstd always output fp32
        meanGm.SetGlobalBuffer((__gm__ float *)mean + block_idx * nlFirstDimPerCore);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + block_idx * nlFirstDimPerCore);
        xGm.SetGlobalBuffer((__gm__ T *)(x) + block_idx * gmOffset_);
    }

    __aicore__ inline void InitUBBuffer()
    {
        Ppipe->InitBuffer(inputOutputQue, BUFFER_NUM, ROUND_UP32(numLastDim * sizeof(T)));
        if constexpr (IS_X_B16_GAMMA_B32) {
            Ppipe->InitBuffer(tmpQueFp32, BUFFER_NUM, ROUND_UP32(numLastDim * sizeof(float)));
        }
        Ppipe->InitBuffer(xBufFp32, ROUND_UP32(numLastDim * sizeof(float)));
        Ppipe->InitBuffer(yBufFp32, ROUND_UP32(numLastDim * sizeof(float)));
#if OUTPUT_MEAN_RSTD == 1
        Ppipe->InitBuffer(meanQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
        Ppipe->InitBuffer(rstdQue, BUFFER_NUM, ROUND_UP32(rowStep * sizeof(float)));
#endif
    }

    __aicore__ inline void Process()
    {
        int32_t rowMoveCnt = CEIL_DIV(rowWork, rowStep);

        for (int32_t rowIdx = 0; rowIdx < rowMoveCnt; ++rowIdx) {
            uint32_t gmOffset = rowIdx * rowStep * numLastDim;
            CopyInAdd(gmOffset, numLastDim);
            if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
                CopyOutX(gmOffset, numLastDim);
            }
            CopyInGammaOneRow();
            ComputeFirstPart();  // compute mean rstd and part of y
            CopyInBetaOneRow();
            ComputeSecondPart();  // compute y
            CopyOut(rowIdx, 1);
        }
    }

private:
    template <typename T_NOCAST, typename T_NEEDCAST>
    __aicore__ inline void CopyInAddWithCast(
        GlobalTensor<T_NOCAST> &xNoCastGm, GlobalTensor<T_NEEDCAST> &xNeedCastGm, uint32_t gmOffset, uint32_t size)
    {
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        auto xBufLocal = xBufFp32.Get<float>();
        auto yBufLocal = yBufFp32.Get<float>();

        // 1. x1/x2 datacopy to ub together and cast
        LocalTensor<T_NOCAST> xNoCastLocalIn = inputOutputQue.template AllocTensor<T_NOCAST>();
        auto tmpLocal = xBufLocal.template ReinterpretCast<T_NEEDCAST>();
        DataCopyEx(tmpLocal, xNeedCastGm[gmOffset], size);

        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
        Cast(yBufLocal, tmpLocal, RoundMode::CAST_NONE, size);  // cast together with MTE2
        DataCopyEx(xNoCastLocalIn, xNoCastGm[gmOffset], size);
        inputOutputQue.EnQue(xNoCastLocalIn);

        // 2. add x1x2
        LocalTensor<T_NOCAST> xNoCastLocal = inputOutputQue.template DeQue<T_NOCAST>();
        pipe_barrier(PIPE_V);
        Add(xBufLocal, yBufLocal, xNoCastLocal, size);
        inputOutputQue.FreeTensor(xNoCastLocal);
    }

    __aicore__ inline void CopyInAddWithoutCast(uint32_t gmOffset, uint32_t size)
    {
        auto xBufLocal = xBufFp32.Get<float>();

        // 1. x1/x2 datacopy to ub together
        LocalTensor<T> xLocalIn = inputOutputQue.template AllocTensor<T>();
        DataCopyEx(xLocalIn, x1Gm[gmOffset], size);
        DataCopyEx(xBufLocal, x2Gm[gmOffset], size);
        inputOutputQue.EnQue(xLocalIn);

        // 2. add x1x2
        LocalTensor<T> xLocal = inputOutputQue.template DeQue<T>();
        Add(xBufLocal, xBufLocal, xLocal, size);
        inputOutputQue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyInAddAllCast(uint32_t gmOffset, uint32_t size)
    {
        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        auto xBufLocal = xBufFp32.Get<float>();

        // 1. x1/x2 datacopy to ub together
        LocalTensor<T> inputLocalIn = inputOutputQue.template AllocTensor<T>();
        LocalTensor<float> tmpLocalIn = tmpQueFp32.template AllocTensor<float>();
        auto tmpLocalInHalf = tmpLocalIn.ReinterpretCast<T>();
        DataCopyEx(inputLocalIn, x1Gm[gmOffset], size);
        DataCopyEx(tmpLocalInHalf, x2Gm[gmOffset], size);
        inputOutputQue.EnQue(inputLocalIn);
        tmpQueFp32.EnQue(tmpLocalIn);

        // 2. fp32 add x1/x2
        LocalTensor<T> inputLocal = inputOutputQue.template DeQue<T>();
        LocalTensor<float> tmpLocal = tmpQueFp32.template DeQue<float>();
        auto tmpLocalHalf = tmpLocal.ReinterpretCast<T>();
        Cast(xBufLocal, tmpLocalHalf, RoundMode::CAST_NONE, size);
        pipe_barrier(PIPE_V);
        Cast(tmpLocal, inputLocal, RoundMode::CAST_NONE, size);
        pipe_barrier(PIPE_V);
        Add(xBufLocal, xBufLocal, tmpLocal, size);
        pipe_barrier(PIPE_V);

        // 3. cast x_sum
        if constexpr (IsSame<T, half>::value) {
            Cast(inputLocal, xBufLocal, RoundMode::CAST_NONE, size);
        } else {
            Cast(inputLocal, xBufLocal, RoundMode::CAST_RINT, size);
        }
        pipe_barrier(PIPE_V);
        inputOutputQue.EnQue(inputLocal);
        tmpQueFp32.FreeTensor(tmpLocal);
    }

    __aicore__ inline void CopyInAddBiasAllCast(uint32_t gmOffset, uint32_t size)
    {
        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        auto xBufLocal = xBufFp32.Get<float>();
        auto yBufLocal = yBufFp32.Get<float>();
        auto xLocalHalf = xBufLocal.ReinterpretCast<T>();
        auto yLocalHalf = yBufLocal.ReinterpretCast<T>();

        // 1. x2/bias datacopy to ub together
        LocalTensor<T> inputLocalIn = inputOutputQue.template AllocTensor<T>();
        LocalTensor<float> tmpLocalIn = tmpQueFp32.template AllocTensor<float>();
        auto tmpLocalInHalf = tmpLocalIn.ReinterpretCast<T>();
        DataCopyEx(inputLocalIn, biasGm, size);
        DataCopyEx(tmpLocalInHalf, x2Gm[gmOffset], size);
        inputOutputQue.EnQue(inputLocalIn);
        tmpQueFp32.EnQue(tmpLocalIn);

        // 2. fp32 add x2/bias
        LocalTensor<T> inputLocal = inputOutputQue.template DeQue<T>();
        LocalTensor<float> tmpLocal = tmpQueFp32.template DeQue<float>();
        auto tmpLocalHalf = tmpLocal.ReinterpretCast<T>();
        Cast(xBufLocal, tmpLocalHalf, RoundMode::CAST_NONE, size);
        pipe_barrier(PIPE_V);
        Cast(tmpLocal, inputLocal, RoundMode::CAST_NONE, size);
        pipe_barrier(PIPE_V);
        Add(xBufLocal, xBufLocal, tmpLocal, size);
        pipe_barrier(PIPE_V);
        inputOutputQue.FreeTensor(inputLocal);
        tmpQueFp32.FreeTensor(tmpLocal);

        // 3. x1 datacopy to ub
        set_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
        DataCopyEx(yLocalHalf, x1Gm[gmOffset], size);
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);

        // 4. fp32 add x1 to x2/bias
        LocalTensor<T> xOutLocal = inputOutputQue.template AllocTensor<T>();
        LocalTensor<float> tmpLocalIn2 = tmpQueFp32.template AllocTensor<float>();
        tmpQueFp32.EnQue(tmpLocalIn2);
        LocalTensor<float> tmpLocal2 = tmpQueFp32.template DeQue<float>();
        Cast(tmpLocal2, yLocalHalf, RoundMode::CAST_NONE, size);
        pipe_barrier(PIPE_V);
        Add(xBufLocal, xBufLocal, tmpLocal2, size);
        pipe_barrier(PIPE_V);
        if constexpr (IsSame<T, half>::value) {
            Cast(xOutLocal, xBufLocal, RoundMode::CAST_NONE, size);
        } else {
            Cast(xOutLocal, xBufLocal, RoundMode::CAST_RINT, size);
        }
        pipe_barrier(PIPE_V);
        inputOutputQue.EnQue(xOutLocal);
    }

    __aicore__ inline void CopyInAdd(uint32_t gmOffset, uint32_t size)
    {
        if constexpr (IS_X1_NEEDCAST) {
            CopyInAddWithCast<T_X2, T_X1>(x2Gm, x1Gm, gmOffset, size);
        } else if constexpr (IS_X2_NEEDCAST) {
            CopyInAddWithCast<T_X1, T_X2>(x1Gm, x2Gm, gmOffset, size);
        } else if constexpr (IS_X1_X2_ALL_FP32) {
            CopyInAddWithoutCast(gmOffset, size);
        } else if constexpr (IS_X_B16_GAMMA_B32 && IS_BIAS_BROADCAST) {
            CopyInAddBiasAllCast(gmOffset, size);
        } else if constexpr (IS_X_B16_GAMMA_B32 && IS_NO_BIAS) {
            CopyInAddAllCast(gmOffset, size);
        }
    }

    __aicore__ inline void CopyOutX(uint32_t gmOffset, uint32_t size)
    {
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        if constexpr (IsSame<T, float>::value) {
            auto addBufLocal = xBufFp32.Get<float>();
            set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
            wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
            DataCopyEx(xGm[gmOffset], addBufLocal, size);
        } else {
            event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            LocalTensor<float> xOutLocal = inputOutputQue.template DeQue<float>();
            auto xOutLocalHalf = xOutLocal.ReinterpretCast<T>();
            set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
            wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
            DataCopyEx(xGm[gmOffset], xOutLocalHalf, size);
            set_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
            wait_flag(PIPE_MTE3, PIPE_MTE2, eventMTE3MTE2);
            inputOutputQue.FreeTensor(xOutLocal);
        }
    }

    __aicore__ inline void CopyInGammaOneRow()
    {
        if constexpr (IsSame<T, float>::value) {  // T_GAMMA and T is float
            LocalTensor<T> gammaLocal = inputOutputQue.template AllocTensor<T>();
            DataCopyEx(gammaLocal, gammaGm, numLastDim);
            inputOutputQue.EnQue(gammaLocal);
        } else if constexpr (IS_X_B16_GAMMA_B32) {  // T_GAMMA is float not equal T
            LocalTensor<float> gammaLocalIn = tmpQueFp32.template AllocTensor<float>();
            DataCopyEx(gammaLocalIn, gammaGm, numLastDim);
            tmpQueFp32.EnQue(gammaLocalIn);
        }
    }

    __aicore__ inline void CopyInBetaOneRow()
    {
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        LocalTensor<T_GAMMA> betaLocal = yBufFp32.Get<float>();

        set_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
        wait_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
        DataCopyEx(betaLocal, betaGm, numLastDim);
    }

    __aicore__ inline void CopyOut(int32_t rowIdx, int32_t row_count)
    {
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

        auto yLocalFp32 = yBufFp32.Get<float>();
        uint32_t gmOffset = rowIdx * rowStep * numLastDim;

        set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        if constexpr (IsSame<T, float>::value) {
            DataCopyEx(yGm[gmOffset], yLocalFp32, numLastDim, row_count);
        } else if constexpr (IS_X_B16_GAMMA_B32) {
            auto yLocalFp32Half = yLocalFp32.ReinterpretCast<T>();
            DataCopyEx(yGm[gmOffset], yLocalFp32Half, numLastDim, row_count);
        }
        set_flag(PIPE_MTE3, PIPE_V, eventMTE3V);
        wait_flag(PIPE_MTE3, PIPE_V, eventMTE3V);

#if OUTPUT_MEAN_RSTD == 1
        uint32_t gm_offset_mean = rowIdx * rowStep;
        LocalTensor<float> mean = meanQue.template DeQue<float>();
        LocalTensor<float> rstd = rstdQue.template DeQue<float>();
        DataCopyEx(meanGm[gm_offset_mean], mean, row_count);
        DataCopyEx(rstdGm[gm_offset_mean], rstd, row_count);
        meanQue.FreeTensor(mean);
        rstdQue.FreeTensor(rstd);
#endif
    }

    __aicore__ inline void ComputeFirstPart()
    {
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> meanLocal = meanQue.template AllocTensor<float>();
        LocalTensor<float> rstdLocal = rstdQue.template AllocTensor<float>();
#endif
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        // 1. mean process: 1/N * x_sum
        Muls(yLocalFp32, xLocalFp32, aveNum, numLastDim);
        pipe_barrier(PIPE_V);
        // 2. mean end: reduce(1/N * x_sum)
        float meanLocalTemp = ReduceSumFP32(yLocalFp32, numLastDim);
        set_flag(PIPE_S, PIPE_V, eventSV);
        wait_flag(PIPE_S, PIPE_V, eventSV);

        // 3. rstd process: x - mean
        Adds(yLocalFp32, xLocalFp32, meanLocalTemp * -1, numLastDim);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_MTE3, PIPE_V, eventMTE3V);  // need make sure xout MTE3 finish.
        wait_flag(PIPE_MTE3, PIPE_V, eventMTE3V);
        // 4. rstd process: (x - mean) ^ 2
        Mul(xLocalFp32, yLocalFp32, yLocalFp32, numLastDim);
        pipe_barrier(PIPE_V);
        // 5. rstd process: reduce( 1 / N * (x - mean) ^ 2 )
        Muls(xLocalFp32, xLocalFp32, aveNum, numLastDim);
        pipe_barrier(PIPE_V);
        float varLocalTemp = ReduceSumFP32(xLocalFp32, numLastDim);
        // 6. rstd end: 1 / sqrt( 1 / N * reduce( (x - mean) ^ 2 ) )
        float rstdLocalTemp = 1 / sqrt(varLocalTemp + eps);

#if OUTPUT_MEAN_RSTD == 1
        meanLocal.SetValue(0, meanLocalTemp);
        rstdLocal.SetValue(0, rstdLocalTemp);
#endif
        set_flag(PIPE_S, PIPE_V, eventSV);
        wait_flag(PIPE_S, PIPE_V, eventSV);
        // 7. y process: (x - mean) / rstd
        Muls(xLocalFp32, yLocalFp32, rstdLocalTemp, numLastDim);
        pipe_barrier(PIPE_V);

#if OUTPUT_MEAN_RSTD == 1
        meanQue.EnQue(meanLocal);
        rstdQue.EnQue(rstdLocal);
#endif
    }

    __aicore__ inline void ComputeSecondPart()
    {
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));

        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> gammaLocal;
        if constexpr (IsSame<T, float>::value) {
            gammaLocal = inputOutputQue.template DeQue<T>();
        } else if constexpr (IS_X_B16_GAMMA_B32) {
            gammaLocal = tmpQueFp32.template DeQue<float>();
        }
        auto yLocalFp32 = yBufFp32.Get<float>();

        Mul(xLocalFp32, xLocalFp32, gammaLocal, numLastDim);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_MTE2, PIPE_V, eventMTE2V);  // unuse deque, need make sure MTE2 finish.
        wait_flag(PIPE_MTE2, PIPE_V, eventMTE2V);
        if constexpr (IsSame<T, float>::value) {
            Add(yLocalFp32, xLocalFp32, yLocalFp32, numLastDim);
        } else if constexpr (IsSame<T, half>::value) {
            Add(xLocalFp32, xLocalFp32, yLocalFp32, numLastDim);
            pipe_barrier(PIPE_V);
            Cast(yLocalFp32.ReinterpretCast<T>(), xLocalFp32, RoundMode::CAST_NONE, numLastDim);
        } else {
            Add(xLocalFp32, xLocalFp32, yLocalFp32, numLastDim);
            pipe_barrier(PIPE_V);
            Cast(yLocalFp32.ReinterpretCast<T>(), xLocalFp32, RoundMode::CAST_RINT, numLastDim);
        }
        pipe_barrier(PIPE_V);

        if constexpr (IS_X_B16_GAMMA_B32) {
            tmpQueFp32.FreeTensor(gammaLocal);
        } else {
            inputOutputQue.FreeTensor(gammaLocal);
        }
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputOutputQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> tmpQueFp32;
#if OUTPUT_MEAN_RSTD == 1
    TQue<QuePosition::VECOUT, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstdQue;
#endif
    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;
    GlobalTensor<T_X1> x1Gm;
    GlobalTensor<T_X2> x2Gm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<T_GAMMA> betaGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> xGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> workspaceGm;
    uint32_t numCore;
    uint32_t numFirstDim;
    uint32_t numLastDim;
    uint32_t rowStep;
    uint32_t rowWork;
    uint32_t rowTail_;
    uint32_t colTail;
    uint32_t colMoveCnt;
    uint32_t firstDimPerTime;
    uint32_t lastDimPerTime;
    uint32_t nlFirstDimPerCore;
    uint32_t lFirstDimPerCore;
    float eps;
    float aveNum;
    bool lastDimPad = false;
    size_t numLastDimAligned;
    bool lastDimPadMixDtype = false;
    size_t numLastDimAlignedMixDtype;
};

template <typename T_X1, typename T_X2, typename T_GAMMA, typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNorm {
#define IS_ADDITIONAL_OUTPUT_ENABLE ((TILING_KEY % 1000) / 100 == 1)
#define IS_NORMAL_CASE ((TILING_KEY % 100) / 10 == 0)
#define IS_SLICE_CASE ((TILING_KEY % 100) / 10 == 1)
#define IS_SINGLE_ROW_CASE ((TILING_KEY % 100) / 10 == 2)
#define IS_SINGLE_ROW_EXT_CASE ((TILING_KEY % 100) / 10 == 3)
#define IS_NORMAL_BIG_N_CASE ((TILING_KEY % 100) / 10 == 4)
#define IS_SLICE_EXT_CASE ((TILING_KEY % 100) / 10 == 5)
#define IS_BIAS_PRESENT ((TILING_KEY % 10) == 1)
#define IS_BIAS_BROADCAST ((TILING_KEY % 10) == 2)
#define IS_CAST_BEFORE_ADD (!IsSame<T_X1, T_X2>::value)
#define IS_X1_NEEDCAST ((!IsSame<T_X1, float>::value) && IS_CAST_BEFORE_ADD)
#define IS_X2_NEEDCAST ((!IsSame<T_X2, float>::value) && IS_CAST_BEFORE_ADD)
#define IS_BETAGAMMA_NEEDCAST (!IsSame<T_GAMMA, float>::value)

public:
    __aicore__ inline KernelAddLayerNorm(TPipe *pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y > 0) {
            return (x + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline uint32_t ROUND_UP32(uint32_t x)
    {
        return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return x < y ? x : y;
    }

    __aicore__ inline uint32_t MAX(uint32_t x, uint32_t y)
    {
        return x > y ? x : y;
    }

    __aicore__ inline void Init(__gm__ uint8_t *x1, __gm__ uint8_t *x2, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
        __gm__ uint8_t *bias, __gm__ uint8_t *y, __gm__ uint8_t *mean, __gm__ uint8_t *rstd, __gm__ uint8_t *x,
        __gm__ uint8_t *workspace, uint32_t num_core_, uint32_t num_Last_dim_, uint32_t num_first_dim_,
        uint32_t nl_first_dim_per_core_, uint32_t l_first_dim_per_core_, uint32_t first_dim_per_time_,
        uint32_t last_dim_per_time_, float eps_, float aveNum_, uint32_t col_move_cnt_, uint32_t col_tail_,
        uint32_t workspace_size)
    {
        num_core = num_core_;
        num_last_dim = num_Last_dim_;
        num_first_dim = num_first_dim_;
        nl_first_dim_per_core = nl_first_dim_per_core_;
        l_first_dim_per_core = l_first_dim_per_core_;
        first_dim_per_time = first_dim_per_time_;
        last_dim_per_time = last_dim_per_time_;
        aveNum = aveNum_;
        eps = eps_;
        col_move_cnt = col_move_cnt_;
        col_tail = col_tail_;
        if (block_idx != num_core - 1) {
            row_work = nl_first_dim_per_core;
            row_step = first_dim_per_time;
        } else {
            row_work = l_first_dim_per_core;
            row_step = MIN(first_dim_per_time, row_work);
        }
        row_tail_ = (row_work % row_step == 0) ? row_step : (row_work % row_step);
        gm_offset_ = nl_first_dim_per_core * num_last_dim;
        x1_gm.SetGlobalBuffer((__gm__ T_X1 *)(x1) + block_idx * gm_offset_);
        x2_gm.SetGlobalBuffer((__gm__ T_X2 *)(x2) + block_idx * gm_offset_);
        if constexpr (IS_BIAS_PRESENT) {
            bias_gm.SetGlobalBuffer((__gm__ T *)(bias) + block_idx * gm_offset_);
        } else if constexpr (IS_BIAS_BROADCAST) {
            bias_gm.SetGlobalBuffer((__gm__ T *)bias);
        }
        gamma_gm.SetGlobalBuffer((__gm__ T_GAMMA *)gamma);
        beta_gm.SetGlobalBuffer((__gm__ T_GAMMA *)beta);
        y_gm.SetGlobalBuffer((__gm__ T *)(y) + block_idx * gm_offset_);
        // mean/rstd always output fp32
        mean_gm.SetGlobalBuffer((__gm__ float *)mean + block_idx * nl_first_dim_per_core);
        rstd_gm.SetGlobalBuffer((__gm__ float *)rstd + block_idx * nl_first_dim_per_core);
        x_gm.SetGlobalBuffer((__gm__ T *)(x) + block_idx * gm_offset_);
        workspace_gm.SetGlobalBuffer((__gm__ float *)workspace + workspace_size);
        num_last_dim_aligned = num_last_dim;
        if (ROUND_UP32(num_last_dim * sizeof(T)) != num_last_dim * sizeof(T)) {
            lastDimPad = true;
            num_last_dim_aligned = ROUND_UP32(num_last_dim * sizeof(T)) / sizeof(T);
        }
        if constexpr (IS_X1_NEEDCAST || IS_X2_NEEDCAST) {
            numLastDimAlignedMixDtype = num_last_dim;
            if (ROUND_UP32(num_last_dim * sizeof(half)) != num_last_dim * sizeof(half)) {
                lastDimPadMixDtype = true;
                numLastDimAlignedMixDtype = ROUND_UP32(num_last_dim * sizeof(half)) / sizeof(half);
            }
        }

        if constexpr (IS_NORMAL_CASE || IS_NORMAL_BIG_N_CASE) {  // normal case
            Ppipe->InitBuffer(x1_que, BUFFER_NUM, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(T)));
            Ppipe->InitBuffer(x2_que, BUFFER_NUM, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(T)));
            Ppipe->InitBuffer(y_que, BUFFER_NUM, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(T)));
            Ppipe->InitBuffer(beta_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(gamma_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            if (IS_NORMAL_BIG_N_CASE && num_last_dim_aligned < ONE_BLK_FLOAT_NUM * 2) {
                Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(row_step * num_last_dim_aligned * 2 * sizeof(float)));
            } else {
                Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(float)));
            }
            Ppipe->InitBuffer(y_buf_fp32, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(float)));
            Ppipe->InitBuffer(z_buf_fp32, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(float)));
            if constexpr (IS_BIAS_BROADCAST) {
                Ppipe->InitBuffer(bias_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            }
        } else if constexpr (IS_SLICE_CASE) {  // slice case
            Ppipe->InitBuffer(x1_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(x2_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(y_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(
                beta_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));  // full load beta/gamma
            Ppipe->InitBuffer(gamma_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(num_last_dim * sizeof(float)));  // store x
            Ppipe->InitBuffer(y_buf_fp32, ROUND_UP32(last_dim_per_time * sizeof(float)));
            Ppipe->InitBuffer(z_buf_fp32, ROUND_UP32(last_dim_per_time * sizeof(float)));
            if constexpr (IS_BIAS_BROADCAST) {
                Ppipe->InitBuffer(bias_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));  // full load bias
            }
        } else if constexpr (IS_SLICE_EXT_CASE) {  // slice ext case
            Ppipe->InitBuffer(x1_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(x2_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(y_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            Ppipe->InitBuffer(beta_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(gamma_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(last_dim_per_time * sizeof(float)));
            Ppipe->InitBuffer(y_buf_fp32, ROUND_UP32(last_dim_per_time * sizeof(float)));
            Ppipe->InitBuffer(z_buf_fp32, ROUND_UP32(last_dim_per_time * sizeof(float)));
            if constexpr (IS_BIAS_BROADCAST) {
                Ppipe->InitBuffer(bias_que, BUFFER_NUM, ROUND_UP32(last_dim_per_time * sizeof(T)));
            }
        } else if constexpr (IS_SINGLE_ROW_CASE) {  // single row
            Ppipe->InitBuffer(x1_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            Ppipe->InitBuffer(x2_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            Ppipe->InitBuffer(y_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            Ppipe->InitBuffer(beta_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(gamma_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(num_last_dim * sizeof(float)));
            Ppipe->InitBuffer(y_buf_fp32, ROUND_UP32(num_last_dim * sizeof(float)));
        } else if constexpr (IS_SINGLE_ROW_EXT_CASE) {  // single row ext case
            Ppipe->InitBuffer(x1_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            Ppipe->InitBuffer(y_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T)));
            Ppipe->InitBuffer(beta_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(gamma_que, BUFFER_NUM, ROUND_UP32(num_last_dim * sizeof(T_GAMMA)));
            Ppipe->InitBuffer(x_buf_fp32, ROUND_UP32(num_last_dim * sizeof(float)));
            Ppipe->InitBuffer(y_buf_fp32, ROUND_UP32(num_last_dim * sizeof(float)));
        }
#if OUTPUT_MEAN_RSTD == 1
        Ppipe->InitBuffer(mean_que, BUFFER_NUM, ROUND_UP32(row_step * sizeof(float)));
        Ppipe->InitBuffer(rstd_que, BUFFER_NUM, ROUND_UP32(row_step * sizeof(float)));
#endif
        if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
            if constexpr (IS_NORMAL_CASE || IS_NORMAL_BIG_N_CASE) {
                Ppipe->InitBuffer(x_que, BUFFER_NUM, ROUND_UP32(row_step * num_last_dim_aligned * sizeof(T)));
            } else if constexpr (!IS_SINGLE_ROW_EXT_CASE) {  // SINGLE_ROW_EXT_CASE x share que with y
                Ppipe->InitBuffer(x_que, BUFFER_NUM, ROUND_UP32(row_step * last_dim_per_time * sizeof(T)));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (IS_NORMAL_CASE || IS_NORMAL_BIG_N_CASE) {
            ProcessNormal();
        } else if constexpr (IS_SLICE_CASE) {
            ProcessSlice();
        } else if constexpr (IS_SINGLE_ROW_CASE || IS_SINGLE_ROW_EXT_CASE) {
            ProcessSingleRow();
        } else if constexpr (IS_SLICE_EXT_CASE) {
            ProcessSliceExt();
        }
    }

    __aicore__ inline void ProcessNormal()
    {
        int32_t row_move_cnt = CEIL_DIV(row_work, row_step);
        CopyInPhase0();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template DeQue<T_GAMMA>();
        LocalTensor<T_GAMMA> beta_local = beta_que.template DeQue<T_GAMMA>();
        LocalTensor<T> bias_local;
        if constexpr (IS_BIAS_BROADCAST) {
            bias_local = bias_que.template DeQue<T>();
        }
        for (int32_t row_idx = 0; row_idx < row_move_cnt; ++row_idx) {
            if (row_idx < row_move_cnt - 1) {
                CopyIn(row_idx, row_step);
                if constexpr (IS_BIAS_PRESENT) {
                    CopyInAddBiasNormal(row_idx, row_step);
                } else if constexpr (IS_BIAS_BROADCAST) {
                    AddBiasBroadCast(row_step, bias_local);
                }
                CopyOutAdditionalOutput(row_idx, row_step);
                if constexpr (IS_NORMAL_BIG_N_CASE) {
                    precision_compute_big_n(row_step, gamma_local, beta_local);
                } else {
                    precision_compute(row_step, gamma_local, beta_local);
                }
                CopyOut(row_idx, row_step);
            } else {
                CopyIn(row_idx, row_tail_);
                if constexpr (IS_BIAS_PRESENT) {
                    CopyInAddBiasNormal(row_idx, row_tail_);
                } else if constexpr (IS_BIAS_BROADCAST) {
                    AddBiasBroadCast(row_tail_, bias_local);
                }
                CopyOutAdditionalOutput(row_idx, row_tail_);
                if constexpr (IS_NORMAL_BIG_N_CASE) {
                    precision_compute_big_n(row_tail_, gamma_local, beta_local);
                } else {
                    precision_compute(row_tail_, gamma_local, beta_local);
                }
                CopyOut(row_idx, row_tail_);
            }
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        if constexpr (IS_BIAS_BROADCAST) {
            bias_que.FreeTensor(bias_local);
        }
    }

    __aicore__ inline void ProcessSingleRow()
    {
        int32_t row_move_cnt = CEIL_DIV(row_work, row_step);
        CopyInPhase0();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template DeQue<T_GAMMA>();
        LocalTensor<T_GAMMA> beta_local = beta_que.template DeQue<T_GAMMA>();
        for (int32_t row_idx = 0; row_idx < row_move_cnt; ++row_idx) {
            CopyInAddSingleRow(row_idx, num_last_dim);
            precision_compute_single_row(gamma_local, beta_local);
            CopyOut(row_idx, 1);
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void ProcessSlice()
    {
        int32_t row_move_cnt = CEIL_DIV(row_work, row_step);
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();

        CopyInSlicePhase0(num_last_dim);
        LocalTensor<T_GAMMA> beta_local = beta_que.template DeQue<T_GAMMA>();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template DeQue<T_GAMMA>();
        LocalTensor<T> bias_local;
        if constexpr (IS_BIAS_BROADCAST) {
            bias_local = bias_que.template DeQue<T>();
        }
        for (int32_t row_idx = 0; row_idx < row_move_cnt; ++row_idx) {
#if OUTPUT_MEAN_RSTD == 1
            LocalTensor<float> mean_local = mean_que.template AllocTensor<float>();
            LocalTensor<float> rstd_local = rstd_que.template AllocTensor<float>();
#endif
            // Reduce Mean
            float ave_tmp = 0;
            Duplicate(z_local_fp32, aveNum, last_dim_per_time);
            for (int32_t col_idx = 0; col_idx < col_move_cnt; ++col_idx) {
                auto col_offset = col_idx * last_dim_per_time;
                int process_count = col_idx < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                CopyInSlicePhase1(row_idx, process_count, col_offset);
                LocalTensor<T> x1_local = x1_que.template DeQue<T>();
                LocalTensor<T> x2_local = x2_que.template DeQue<T>();
                if constexpr (IsSame<T, float>::value) {
                    if constexpr (IS_X1_NEEDCAST) {
                        auto y_local_buffer = y_local_fp32.template ReinterpretCast<T_X1>();
                        Cast(x1_local, y_local_buffer, RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                    }
                    if constexpr (IS_X2_NEEDCAST) {
                        auto y_local_buffer = y_local_fp32.template ReinterpretCast<T_X2>();
                        Cast(x2_local, y_local_buffer, RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                    }
                    Add(x_local_fp32[col_offset], x1_local, x2_local, process_count);
                    pipe_barrier(PIPE_V);
                } else {
                    Cast(x_local_fp32[col_offset], x1_local, RoundMode::CAST_NONE, process_count);
                    Cast(y_local_fp32, x2_local, RoundMode::CAST_NONE, process_count);
                    pipe_barrier(PIPE_V);
                    Add(x_local_fp32[col_offset], x_local_fp32[col_offset], y_local_fp32, process_count);
                    pipe_barrier(PIPE_V);
                }
                x1_que.FreeTensor(x1_local);
                x2_que.FreeTensor(x2_local);

                if constexpr (IS_BIAS_PRESENT) {
                    LocalTensor<T> x3_in = x1_que.template AllocTensor<T>();
                    uint32_t gm_offset = row_idx * row_step * num_last_dim + col_offset;
                    DataCopyEx(x3_in, bias_gm[gm_offset], process_count);
                    x1_que.EnQue(x3_in);
                    auto x3_local = x1_que.template DeQue<T>();
                    if constexpr (IsSame<T, float>::value) {
                        Add(x_local_fp32[col_offset], x3_local, x_local_fp32[col_offset], process_count);
                        pipe_barrier(PIPE_V);
                    } else {
                        Cast(y_local_fp32, x3_local, RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                        Add(x_local_fp32[col_offset], y_local_fp32, x_local_fp32[col_offset], process_count);
                        pipe_barrier(PIPE_V);
                    }
                    x1_que.FreeTensor(x3_local);
                } else if constexpr (IS_BIAS_BROADCAST) {
                    if constexpr (IsSame<T, float>::value) {
                        Add(x_local_fp32[col_offset], bias_local[col_offset], x_local_fp32[col_offset], process_count);
                        pipe_barrier(PIPE_V);
                    } else {
                        Cast(y_local_fp32, bias_local[col_offset], RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                        Add(x_local_fp32[col_offset], y_local_fp32, x_local_fp32[col_offset], process_count);
                        pipe_barrier(PIPE_V);
                    }
                }

                if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
                    LocalTensor<T> x_local = x_que.template AllocTensor<T>();
                    if constexpr (IsSame<T, float>::value) {
                        Adds(x_local, x_local_fp32[col_offset], ZERO, process_count);
                    } else if constexpr (IsSame<T, half>::value) {
                        Cast(x_local, x_local_fp32[col_offset], RoundMode::CAST_NONE, process_count);
                    } else {
                        Cast(x_local, x_local_fp32[col_offset], RoundMode::CAST_RINT, process_count);
                    }
                    x_que.EnQue(x_local);
                    CopyOutSlicePhase0(row_idx, process_count, last_dim_per_time * col_idx);
                }
                Mul(y_local_fp32, x_local_fp32[col_offset], z_local_fp32, process_count);
                pipe_barrier(PIPE_V);
                ReduceSum(y_local_fp32, y_local_fp32, y_local_fp32, process_count);
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                ave_tmp += y_local_fp32.GetValue(0);
            }
            // 2. Reduce Var
            float var_tmp = 0;
            for (int32_t col_idx = 0; col_idx < col_move_cnt; ++col_idx) {
                auto col_offset = col_idx * last_dim_per_time;
                int process_count = col_idx < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                Adds(x_local_fp32[col_offset], x_local_fp32[col_offset], -1 * ave_tmp, process_count);
                pipe_barrier(PIPE_V);
                Mul(y_local_fp32, x_local_fp32[col_offset], x_local_fp32[col_offset], process_count);
                pipe_barrier(PIPE_V);
                Muls(y_local_fp32, y_local_fp32, aveNum, process_count);
                pipe_barrier(PIPE_V);
                ReduceSum(y_local_fp32, y_local_fp32, y_local_fp32, process_count);
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                var_tmp += y_local_fp32.GetValue(0);
            }
            float rstd_tmp = 1 / sqrt(var_tmp + eps);
#if OUTPUT_MEAN_RSTD == 1
            mean_local.SetValue(0, ave_tmp);
            rstd_local.SetValue(0, rstd_tmp);
#endif
            // 3. Compute result
            for (int32_t col_idx = 0; col_idx < col_move_cnt; ++col_idx) {
                auto col_offset = col_idx * last_dim_per_time;
                int process_count = col_idx < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                LocalTensor<T> y_local = y_que.template AllocTensor<T>();
                // x_local_fp32[col_offset] = (x - ave)
                Muls(y_local_fp32, x_local_fp32[col_offset], rstd_tmp, process_count);
                pipe_barrier(PIPE_V);
                if constexpr (IS_BETAGAMMA_NEEDCAST) {
                    if constexpr (IsSame<T, float>::value) {
                        Cast(x_local_fp32[col_offset], gamma_local[col_offset], RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                        Mul(y_local_fp32, x_local_fp32[col_offset], y_local_fp32, process_count);
                        Cast(x_local_fp32[col_offset], beta_local[col_offset], RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                        Add(y_local, y_local_fp32, x_local_fp32[col_offset], process_count);
                        pipe_barrier(PIPE_V);
                    } else {
                        Cast(x_local_fp32[col_offset], gamma_local[col_offset], RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                        Mul(y_local_fp32, x_local_fp32[col_offset], y_local_fp32, process_count);
                        Cast(x_local_fp32[col_offset], beta_local[col_offset], RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                        Add(y_local_fp32, y_local_fp32, x_local_fp32[col_offset], process_count);
                        pipe_barrier(PIPE_V);
                        if constexpr (IsSame<T, half>::value) {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_NONE, process_count);
                        } else {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_RINT, process_count);
                        }
                    }
                } else {
                    if constexpr (IsSame<T, float>::value) {
                        Mul(y_local_fp32, gamma_local[col_offset], y_local_fp32, process_count);
                        pipe_barrier(PIPE_V);
                        Add(y_local, y_local_fp32, beta_local[col_offset], process_count);
                    } else {
                        Mul(y_local_fp32, gamma_local[col_offset], y_local_fp32, process_count);
                        pipe_barrier(PIPE_V);
                        Add(y_local_fp32, y_local_fp32, beta_local[col_offset], process_count);
                        pipe_barrier(PIPE_V);
                        if constexpr (IsSame<T, half>::value) {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_NONE, process_count);
                        } else {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_RINT, process_count);
                        }
                    }
                }

                y_que.EnQue(y_local);
                CopyOutSlicePhase1(row_idx, process_count, last_dim_per_time * col_idx);
            }
#if OUTPUT_MEAN_RSTD == 1
            mean_que.EnQue(mean_local);
            rstd_que.EnQue(rstd_local);
            CopyOutSlicePhase2(row_idx);
#endif
        }
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
    }

    __aicore__ inline void CopyInAddSlice(int32_t row_idx, int32_t process_count, int32_t col_offset)
    {
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        CopyInSlicePhase1(row_idx, process_count, col_offset);
        LocalTensor<T> x1_local = x1_que.template DeQue<T>();
        LocalTensor<T> x2_local = x2_que.template DeQue<T>();
        if constexpr (IsSame<T, float>::value) {
            if constexpr (IS_X1_NEEDCAST) {
                auto y_local_buffer = y_local_fp32.template ReinterpretCast<T_X1>();
                Cast(x1_local, y_local_buffer, RoundMode::CAST_NONE, process_count);
                pipe_barrier(PIPE_V);
            }
            if constexpr (IS_X2_NEEDCAST) {
                auto y_local_buffer = y_local_fp32.template ReinterpretCast<T_X2>();
                Cast(x2_local, y_local_buffer, RoundMode::CAST_NONE, process_count);
                pipe_barrier(PIPE_V);
            }
            Add(x_local_fp32, x1_local, x2_local, process_count);
            pipe_barrier(PIPE_V);
        } else {
            Cast(x_local_fp32, x1_local, RoundMode::CAST_NONE, process_count);
            Cast(y_local_fp32, x2_local, RoundMode::CAST_NONE, process_count);
            pipe_barrier(PIPE_V);
            Add(x_local_fp32, x_local_fp32, y_local_fp32, process_count);
            pipe_barrier(PIPE_V);
        }
        x1_que.FreeTensor(x1_local);
        x2_que.FreeTensor(x2_local);
        if constexpr (IS_BIAS_PRESENT) {
            LocalTensor<T> x3_in = x1_que.template AllocTensor<T>();
            uint32_t gm_offset = row_idx * row_step * num_last_dim + col_offset;
            DataCopyEx(x3_in, bias_gm[gm_offset], process_count);
            x1_que.EnQue(x3_in);
            auto x3_local = x1_que.template DeQue<T>();
            if constexpr (IsSame<T, float>::value) {
                Add(x_local_fp32, x3_local, x_local_fp32, process_count);
                pipe_barrier(PIPE_V);
            } else {
                Cast(y_local_fp32, x3_local, RoundMode::CAST_NONE, process_count);
                pipe_barrier(PIPE_V);
                Add(x_local_fp32, y_local_fp32, x_local_fp32, process_count);
                pipe_barrier(PIPE_V);
            }
            x1_que.FreeTensor(x3_local);
        } else if constexpr (IS_BIAS_BROADCAST) {
            LocalTensor<T> bias_local_in = bias_que.template AllocTensor<T>();
            DataCopyEx(bias_local_in, bias_gm[col_offset], process_count);
            bias_que.EnQue(bias_local_in);
            auto bias_local = bias_que.template DeQue<T>();
            if constexpr (IsSame<T, float>::value) {
                Add(x_local_fp32, bias_local, x_local_fp32, process_count);
                pipe_barrier(PIPE_V);
            } else {
                Cast(y_local_fp32, bias_local, RoundMode::CAST_NONE, process_count);
                pipe_barrier(PIPE_V);
                Add(x_local_fp32, y_local_fp32, x_local_fp32, process_count);
                pipe_barrier(PIPE_V);
            }
            bias_que.FreeTensor(bias_local);
        }
    }

    __aicore__ inline void CopyOutAdditionalOutputSlice(int32_t row_idx, int32_t process_count, int32_t col_offset)
    {
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<T> x_local = x_que.template AllocTensor<T>();
        if constexpr (IsSame<T, float>::value) {
            Adds(x_local, x_local_fp32, ZERO, process_count);
        } else if constexpr (IsSame<T, half>::value) {
            Cast(x_local, x_local_fp32, RoundMode::CAST_NONE, process_count);
        } else {
            Cast(x_local, x_local_fp32, RoundMode::CAST_RINT, process_count);
        }
        x_que.EnQue(x_local);
        CopyOutSlicePhase0(row_idx, process_count, col_offset);
    }

    __aicore__ inline void ProcessSliceExt()
    {
        int32_t row_move_cnt = CEIL_DIV(row_work, row_step);
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        for (int32_t row_idx = 0; row_idx < row_move_cnt; ++row_idx) {
#if OUTPUT_MEAN_RSTD == 1
            LocalTensor<float> mean_local = mean_que.template AllocTensor<float>();
            LocalTensor<float> rstd_local = rstd_que.template AllocTensor<float>();
#endif
            // Reduce Mean
            float ave_tmp = 0;
            for (int32_t col_idx = 0; col_idx < col_move_cnt; ++col_idx) {
                auto col_offset = col_idx * last_dim_per_time;
                int process_count = col_idx < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                CopyInAddSlice(row_idx, process_count, col_offset);
                Muls(y_local_fp32, x_local_fp32, aveNum, process_count);
                pipe_barrier(PIPE_V);
                ReduceSum(y_local_fp32, y_local_fp32, y_local_fp32, process_count);
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                ave_tmp += y_local_fp32.GetValue(0);
            }
            // 2. Reduce Var
            float var_tmp = 0;
            for (int32_t col_idx = 0; col_idx < col_move_cnt; ++col_idx) {
                auto col_offset = col_idx * last_dim_per_time;
                int process_count = col_idx < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                CopyInAddSlice(row_idx, process_count, col_offset);
                Adds(x_local_fp32, x_local_fp32, -1 * ave_tmp, process_count);
                pipe_barrier(PIPE_V);
                Mul(y_local_fp32, x_local_fp32, x_local_fp32, process_count);
                pipe_barrier(PIPE_V);
                Muls(y_local_fp32, y_local_fp32, aveNum, process_count);
                pipe_barrier(PIPE_V);
                ReduceSum(y_local_fp32, y_local_fp32, y_local_fp32, process_count);
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                var_tmp += y_local_fp32.GetValue(0);
            }
            float rstd_tmp = 1 / sqrt(var_tmp + eps);
#if OUTPUT_MEAN_RSTD == 1
            mean_local.SetValue(0, ave_tmp);
            rstd_local.SetValue(0, rstd_tmp);
            mean_que.EnQue(mean_local);
            rstd_que.EnQue(rstd_local);
            CopyOutSlicePhase2(row_idx);
#endif
            // 3. Compute result
            for (int32_t col_idx = 0; col_idx < col_move_cnt; ++col_idx) {
                auto col_offset = col_idx * last_dim_per_time;
                int process_count = col_idx < col_move_cnt - 1 ? last_dim_per_time : col_tail;
                CopyInAddSlice(row_idx, process_count, col_offset);
                if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
                    CopyOutAdditionalOutputSlice(row_idx, process_count, col_offset);
                }
                CopyInSlicePhase2(process_count, col_offset);
                LocalTensor<T_GAMMA> beta_local = beta_que.template DeQue<T_GAMMA>();
                LocalTensor<T_GAMMA> gamma_local = gamma_que.template DeQue<T_GAMMA>();
                LocalTensor<T> y_local = y_que.template AllocTensor<T>();
                Adds(x_local_fp32, x_local_fp32, -1 * ave_tmp, process_count);
                pipe_barrier(PIPE_V);
                Muls(y_local_fp32, x_local_fp32, rstd_tmp, process_count);
                pipe_barrier(PIPE_V);

                if constexpr (IS_BETAGAMMA_NEEDCAST) {
                    if constexpr (IsSame<T, float>::value) {
                        Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                        Mul(y_local_fp32, x_local_fp32, y_local_fp32, process_count);
                        Cast(x_local_fp32, beta_local, RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                        Add(y_local, y_local_fp32, x_local_fp32, process_count);
                    } else {
                        Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                        Mul(y_local_fp32, x_local_fp32, y_local_fp32, process_count);
                        Cast(x_local_fp32, beta_local, RoundMode::CAST_NONE, process_count);
                        pipe_barrier(PIPE_V);
                        Add(y_local_fp32, y_local_fp32, x_local_fp32, process_count);
                        pipe_barrier(PIPE_V);
                        if constexpr (IsSame<T, half>::value) {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_NONE, process_count);
                        } else {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_RINT, process_count);
                        }
                    }
                } else {
                    if constexpr (IsSame<T, float>::value) {
                        Mul(y_local_fp32, gamma_local, y_local_fp32, process_count);
                        pipe_barrier(PIPE_V);
                        Add(y_local, y_local_fp32, beta_local, process_count);
                    } else {
                        Mul(y_local_fp32, gamma_local, y_local_fp32, process_count);
                        pipe_barrier(PIPE_V);
                        Add(y_local_fp32, y_local_fp32, beta_local, process_count);
                        pipe_barrier(PIPE_V);
                        if constexpr (IsSame<T, half>::value) {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_NONE, process_count);
                        } else {
                            Cast(y_local, y_local_fp32, RoundMode::CAST_RINT, process_count);
                        }
                    }
                }

                beta_que.FreeTensor(beta_local);
                gamma_que.FreeTensor(gamma_local);
                y_que.EnQue(y_local);
                CopyOutSlicePhase1(row_idx, process_count, last_dim_per_time * col_idx);
            }
        }
    }

private:
    __aicore__ inline void CopyInSlicePhase1(int32_t row_idx, int32_t size, int32_t offset = 0)
    {
        event_t event_mte3_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));

        LocalTensor<T> x1_local = x1_que.template AllocTensor<T>();
        LocalTensor<T> x2_local = x2_que.template AllocTensor<T>();
        uint32_t gm_offset = row_idx * row_step * num_last_dim + offset;
        if constexpr (IS_X1_NEEDCAST) {
            set_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
            wait_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
            LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
            auto y_local_buffer = y_local_fp32.template ReinterpretCast<T_X1>();
            DataCopyEx(y_local_buffer, x1_gm[gm_offset], size);
        } else {
            DataCopyEx(x1_local, x1_gm[gm_offset], size);
        }
        if constexpr (IS_X2_NEEDCAST) {
            set_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
            wait_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
            LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
            auto y_local_buffer = y_local_fp32.template ReinterpretCast<T_X2>();
            DataCopyEx(y_local_buffer, x2_gm[gm_offset], size);
        } else {
            DataCopyEx(x2_local, x2_gm[gm_offset], size);
        }
        x1_que.EnQue(x1_local);
        x2_que.EnQue(x2_local);
    }

    __aicore__ inline void CopyInSlicePhase2(int32_t size, int32_t offset = 0)
    {
        LocalTensor<T_GAMMA> beta_local = beta_que.template AllocTensor<T_GAMMA>();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template AllocTensor<T_GAMMA>();
        DataCopyEx(beta_local, beta_gm[offset], size);
        DataCopyEx(gamma_local, gamma_gm[offset], size);
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
    }

    __aicore__ inline void CopyInSlicePhase0(int32_t size, int32_t offset = 0)
    {
        LocalTensor<T_GAMMA> beta_local = beta_que.template AllocTensor<T_GAMMA>();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template AllocTensor<T_GAMMA>();
        DataCopyEx(beta_local, beta_gm[offset], size);
        DataCopyEx(gamma_local, gamma_gm[offset], size);
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
        if constexpr (IS_BIAS_BROADCAST) {
            LocalTensor<T> bias_local = bias_que.template AllocTensor<T>();
            DataCopyEx(bias_local, bias_gm[offset], size);
            bias_que.EnQue(bias_local);
        }
    }

    __aicore__ inline void CopyInPhase0()
    {
        LocalTensor<T_GAMMA> beta_local = beta_que.template AllocTensor<T_GAMMA>();
        LocalTensor<T_GAMMA> gamma_local = gamma_que.template AllocTensor<T_GAMMA>();
        DataCopyEx(beta_local, beta_gm, num_last_dim);
        DataCopyEx(gamma_local, gamma_gm, num_last_dim);
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
        if constexpr (IS_BIAS_BROADCAST && (!(IS_SINGLE_ROW_CASE || IS_SINGLE_ROW_EXT_CASE))) {
            LocalTensor<T> bias_local = bias_que.template AllocTensor<T>();
            DataCopyEx(bias_local, bias_gm, num_last_dim);
            bias_que.EnQue(bias_local);
        }
    }

    __aicore__ inline void CopyIn(int32_t proc_id, int32_t row_count)
    {
        event_t event_mte3_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_buf_local = y_buf_fp32.Get<float>();
        LocalTensor<float> add_buf_local = z_buf_fp32.Get<float>();
        uint32_t gm_offset = proc_id * row_step * num_last_dim;
        auto elementCount = num_last_dim_aligned * row_count;
        DataCopyPadParams padParams;
        if (lastDimPad) {
            padParams.isPad = true;
            padParams.paddingValue = 0;
            padParams.rightPadding = num_last_dim_aligned - num_last_dim;
        }
        LocalTensor<T> x1_local_in = x1_que.template AllocTensor<T>();
        if constexpr (IS_X1_NEEDCAST) {
            DataCopyPadParams padParamsFp16;
            if (lastDimPadMixDtype) {
                padParamsFp16.isPad = true;
                padParamsFp16.paddingValue = 0;
                padParamsFp16.rightPadding = numLastDimAlignedMixDtype - num_last_dim;
            }
            set_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
            wait_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
            auto y_buf = y_buf_local.template ReinterpretCast<T_X1>();
            DataCopyEx(y_buf, x1_gm[gm_offset], num_last_dim, row_count, padParamsFp16);
        } else {
            DataCopyEx(x1_local_in, x1_gm[gm_offset], num_last_dim, row_count, padParams);
        }
        x1_que.EnQue(x1_local_in);
        auto x1_local = x1_que.template DeQue<T>();
        if constexpr (IsSame<float, T>::value) {
            if constexpr (IS_X1_NEEDCAST) {
                auto y_buf = y_buf_local.template ReinterpretCast<T_X1>();
                for (uint32_t i = 0; i < row_count; i++) {
                    Cast(add_buf_local[i * num_last_dim_aligned],
                        y_buf[i * numLastDimAlignedMixDtype],
                        RoundMode::CAST_NONE,
                        num_last_dim);
                }
            } else {
                Adds(add_buf_local, x1_local, ZERO, elementCount);
            }
            pipe_barrier(PIPE_V);
        } else {
            Cast(add_buf_local, x1_local, RoundMode::CAST_NONE, elementCount);
        }
        x1_que.FreeTensor(x1_local);

        LocalTensor<T> x2_local_in = x2_que.template AllocTensor<T>();
        if constexpr (IS_X2_NEEDCAST) {
            DataCopyPadParams padParamsFp16;
            if (lastDimPadMixDtype) {
                padParamsFp16.isPad = true;
                padParamsFp16.paddingValue = 0;
                padParamsFp16.rightPadding = numLastDimAlignedMixDtype - num_last_dim;
            }
            set_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
            wait_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
            auto y_buf = x_local_fp32.template ReinterpretCast<T_X2>();
            DataCopyEx(y_buf, x2_gm[gm_offset], num_last_dim, row_count, padParamsFp16);
        } else {
            DataCopyEx(x2_local_in, x2_gm[gm_offset], num_last_dim, row_count, padParams);
        }
        x2_que.EnQue(x2_local_in);
        auto x2_local = x2_que.template DeQue<T>();
        if constexpr (IsSame<float, T>::value) {
            if constexpr (IS_X2_NEEDCAST) {
                auto y_buf = x_local_fp32.template ReinterpretCast<T_X2>();
                for (uint32_t i = 0; i < row_count; i++) {
                    Cast(x2_local[i * num_last_dim_aligned],
                        y_buf[i * numLastDimAlignedMixDtype],
                        RoundMode::CAST_NONE,
                        num_last_dim);
                }
                pipe_barrier(PIPE_V);
                Add(add_buf_local, x2_local, add_buf_local, elementCount);
                pipe_barrier(PIPE_V);
            } else {
                Add(add_buf_local, x2_local, add_buf_local, elementCount);
                pipe_barrier(PIPE_V);
            }
        } else {
            Cast(y_buf_local, x2_local, RoundMode::CAST_NONE, elementCount);
            pipe_barrier(PIPE_V);
            Add(add_buf_local, y_buf_local, add_buf_local, elementCount);
            pipe_barrier(PIPE_V);
        }
        x2_que.FreeTensor(x2_local);
    }

    __aicore__ inline void CopyOutAdditionalOutput(int32_t proc_id, int32_t row_count)
    {
        if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
            LocalTensor<float> add_buf_local = z_buf_fp32.Get<float>();
            uint32_t gm_offset = proc_id * row_step * num_last_dim;
            auto elementCount = num_last_dim_aligned * row_count;
            auto x_local = x_que.template AllocTensor<T>();
            if constexpr (IsSame<T, float>::value) {
                Adds(x_local, add_buf_local, ZERO, elementCount);
            } else if constexpr (IsSame<T, half>::value) {
                Cast(x_local, add_buf_local, RoundMode::CAST_NONE, elementCount);
            } else {
                Cast(x_local, add_buf_local, RoundMode::CAST_RINT, elementCount);
            }
            pipe_barrier(PIPE_V);
            x_que.template EnQue<T>(x_local);

            auto x = x_que.template DeQue<T>();
            DataCopyEx(x_gm[gm_offset], x, num_last_dim, row_count);
            x_que.FreeTensor(x);
        }
    }

    __aicore__ inline void AddBiasBroadCast(int32_t row_count, const LocalTensor<T> &bias_local)
    {
        auto y_buf_local = y_buf_fp32.Get<float>();
        auto add_buf_local = z_buf_fp32.Get<float>();
        if constexpr (IsSame<float, T>::value) {
            for (int i = 0; i < row_count; i++) {
                Add(add_buf_local[i * num_last_dim_aligned],
                    bias_local,
                    add_buf_local[i * num_last_dim_aligned],
                    num_last_dim);
                pipe_barrier(PIPE_V);
            }
        } else {
            Cast(y_buf_local, bias_local, RoundMode::CAST_NONE, num_last_dim);
            pipe_barrier(PIPE_V);
            for (int i = 0; i < row_count; i++) {
                Add(add_buf_local[i * num_last_dim_aligned],
                    y_buf_local,
                    add_buf_local[i * num_last_dim_aligned],
                    num_last_dim);
                pipe_barrier(PIPE_V);
            }
        }
    }

    __aicore__ inline void CopyInAddBiasNormal(int32_t proc_id, int32_t row_count)
    {
        auto y_buf_local = y_buf_fp32.Get<float>();
        auto add_buf_local = z_buf_fp32.Get<float>();
        uint32_t gm_offset = proc_id * row_step * num_last_dim;
        auto elementCount = num_last_dim_aligned * row_count;
        DataCopyPadParams padParams;
        if (lastDimPad) {
            padParams.isPad = true;
            padParams.paddingValue = 0;
            padParams.rightPadding = num_last_dim_aligned - num_last_dim;
        }
        LocalTensor<T> x3_local_in = x1_que.template AllocTensor<T>();
        DataCopyEx(x3_local_in, bias_gm[gm_offset], num_last_dim, row_count, padParams);
        x1_que.EnQue(x3_local_in);
        auto x3_local = x1_que.template DeQue<T>();
        if constexpr (IsSame<float, T>::value) {
            Add(add_buf_local, x3_local, add_buf_local, elementCount);
            pipe_barrier(PIPE_V);
        } else {
            Cast(y_buf_local, x3_local, RoundMode::CAST_NONE, elementCount);
            pipe_barrier(PIPE_V);
            Add(add_buf_local, y_buf_local, add_buf_local, elementCount);
            pipe_barrier(PIPE_V);
        }
        x1_que.FreeTensor(x3_local);
    }

    __aicore__ inline void CopyInAddSingleRow(int32_t row_idx, int32_t size)
    {
        event_t event_mte3_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        auto add_buf_local = x_buf_fp32.Get<float>();
        auto y_buf_local = y_buf_fp32.Get<float>();
        uint32_t gm_offset = row_idx * row_step * num_last_dim;
        LocalTensor<T> x1_local_in = x1_que.template AllocTensor<T>();
        if constexpr (IS_X1_NEEDCAST) {
            set_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
            wait_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
            auto y_local_buffer = y_buf_local.template ReinterpretCast<T_X1>();
            DataCopyEx(y_local_buffer, x1_gm[gm_offset], size);
        } else {
            DataCopyEx(x1_local_in, x1_gm[gm_offset], size);
        }
        x1_que.EnQue(x1_local_in);
        auto x1_local = x1_que.template DeQue<T>();
        if constexpr (IsSame<float, T>::value) {
            if constexpr (IS_X1_NEEDCAST) {
                auto y_local_buffer = y_buf_local.template ReinterpretCast<T_X1>();
                Cast(add_buf_local, y_local_buffer, RoundMode::CAST_NONE, size);
            } else {
                Adds(add_buf_local, x1_local, ZERO, size);
            }
            pipe_barrier(PIPE_V);
        } else {
            Cast(add_buf_local, x1_local, RoundMode::CAST_NONE, size);
        }
        x1_que.FreeTensor(x1_local);

        if constexpr (IS_SINGLE_ROW_EXT_CASE) {
            LocalTensor<T> x2_local_in = x1_que.template AllocTensor<T>();
            if constexpr (IS_X2_NEEDCAST) {
                set_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
                wait_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
                auto y_local_buffer = y_buf_local.template ReinterpretCast<T_X2>();
                DataCopyEx(y_local_buffer, x2_gm[gm_offset], size);
            } else {
                DataCopyEx(x2_local_in, x2_gm[gm_offset], size);
            }
            x1_que.EnQue(x2_local_in);
            auto x2_local = x1_que.template DeQue<T>();
            if constexpr (IsSame<float, T>::value) {
                if constexpr (IS_X2_NEEDCAST) {
                    auto y_local_buffer = y_buf_local.template ReinterpretCast<T_X2>();
                    Cast(x2_local, y_local_buffer, RoundMode::CAST_NONE, size);
                    pipe_barrier(PIPE_V);
                }
                Add(add_buf_local, x2_local, add_buf_local, size);
                pipe_barrier(PIPE_V);
            } else {
                Cast(y_buf_local, x2_local, RoundMode::CAST_NONE, size);
                pipe_barrier(PIPE_V);
                Add(add_buf_local, y_buf_local, add_buf_local, size);
                pipe_barrier(PIPE_V);
            }
            x1_que.FreeTensor(x2_local);
        } else {
            LocalTensor<T> x2_local_in = x2_que.template AllocTensor<T>();
            if constexpr (IS_X2_NEEDCAST) {
                set_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
                wait_flag(PIPE_MTE3, PIPE_S, event_mte3_s);
                auto y_local_buffer = y_buf_local.template ReinterpretCast<T_X2>();
                DataCopyEx(y_local_buffer, x2_gm[gm_offset], size);
            } else {
                DataCopyEx(x2_local_in, x2_gm[gm_offset], size);
            }
            x2_que.EnQue(x2_local_in);
            auto x2_local = x2_que.template DeQue<T>();
            if constexpr (IsSame<float, T>::value) {
                if constexpr (IS_X2_NEEDCAST) {
                    auto y_local_buffer = y_buf_local.template ReinterpretCast<T_X2>();
                    Cast(x2_local, y_local_buffer, RoundMode::CAST_NONE, size);
                    pipe_barrier(PIPE_V);
                }
                Add(add_buf_local, x2_local, add_buf_local, size);
                pipe_barrier(PIPE_V);
            } else {
                Cast(y_buf_local, x2_local, RoundMode::CAST_NONE, size);
                pipe_barrier(PIPE_V);
                Add(add_buf_local, y_buf_local, add_buf_local, size);
                pipe_barrier(PIPE_V);
            }
            x2_que.FreeTensor(x2_local);
        }

        if constexpr (IS_BIAS_PRESENT || IS_BIAS_BROADCAST) {
            LocalTensor<T> x3_local_in = x1_que.template AllocTensor<T>();
            if constexpr (IS_BIAS_PRESENT) {
                DataCopyEx(x3_local_in, bias_gm[gm_offset], size);
            } else if constexpr (IS_BIAS_BROADCAST) {
                DataCopyEx(x3_local_in, bias_gm, size);
            }
            x1_que.EnQue(x3_local_in);
            auto x3_local = x1_que.template DeQue<T>();
            if constexpr (IsSame<float, T>::value) {
                Add(add_buf_local, x3_local, add_buf_local, size);
                pipe_barrier(PIPE_V);
            } else {
                Cast(y_buf_local, x3_local, RoundMode::CAST_NONE, size);
                pipe_barrier(PIPE_V);
                Add(add_buf_local, y_buf_local, add_buf_local, size);
                pipe_barrier(PIPE_V);
            }
            x1_que.FreeTensor(x3_local);
        }

        if constexpr (IS_ADDITIONAL_OUTPUT_ENABLE) {
            if constexpr (IS_SINGLE_ROW_EXT_CASE) {
                auto x_local = y_que.template AllocTensor<T>();
                if constexpr (IsSame<T, float>::value) {
                    Adds(x_local, add_buf_local, ZERO, size);
                } else if constexpr (IsSame<T, half>::value) {
                    Cast(x_local, add_buf_local, RoundMode::CAST_NONE, size);
                } else {
                    Cast(x_local, add_buf_local, RoundMode::CAST_RINT, size);
                }
                pipe_barrier(PIPE_V);
                y_que.template EnQue<T>(x_local);

                auto x = y_que.template DeQue<T>();
                DataCopyEx(x_gm[gm_offset], x, size);
                y_que.FreeTensor(x);
            } else {
                auto x_local = x_que.template AllocTensor<T>();
                if constexpr (IsSame<T, float>::value) {
                    Adds(x_local, add_buf_local, ZERO, size);
                } else if constexpr (IsSame<T, half>::value) {
                    Cast(x_local, add_buf_local, RoundMode::CAST_NONE, size);
                } else {
                    Cast(x_local, add_buf_local, RoundMode::CAST_RINT, size);
                }
                pipe_barrier(PIPE_V);
                x_que.template EnQue<T>(x_local);

                auto x = x_que.template DeQue<T>();
                DataCopyEx(x_gm[gm_offset], x, size);
                x_que.FreeTensor(x);
            }
        }
    }

    __aicore__ inline void precision_compute(
        int32_t nums, LocalTensor<T_GAMMA> &gamma_local, LocalTensor<T_GAMMA> &beta_local)
    {
        LocalTensor<T> y_local = y_que.template AllocTensor<T>();
#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> mean_local = mean_que.template AllocTensor<float>();
        LocalTensor<float> rstd_local = rstd_que.template AllocTensor<float>();
#endif
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * num_last_dim_aligned;
            Muls(y_local_fp32, z_local_fp32[roundOffset], aveNum, num_last_dim);
            pipe_barrier(PIPE_V);
            auto ave_local_temp = ReduceSumFP32(y_local_fp32, num_last_dim);
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            Adds(z_local_fp32[roundOffset], z_local_fp32[roundOffset], ave_local_temp * -1, num_last_dim);
            pipe_barrier(PIPE_V);
            Mul(x_local_fp32, z_local_fp32[roundOffset], z_local_fp32[roundOffset], num_last_dim);
            pipe_barrier(PIPE_V);
            Muls(y_local_fp32, x_local_fp32, aveNum, num_last_dim);
            pipe_barrier(PIPE_V);
            float var_local_temp = ReduceSumFP32(y_local_fp32, num_last_dim);
            float rstd_local_temp = 1 / sqrt(var_local_temp + eps);
#if OUTPUT_MEAN_RSTD == 1
            mean_local.SetValue(rid, ave_local_temp);
            rstd_local.SetValue(rid, rstd_local_temp);
#endif
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            Muls(x_local_fp32, z_local_fp32[roundOffset], rstd_local_temp, num_last_dim);
            pipe_barrier(PIPE_V);
            if constexpr (IS_BETAGAMMA_NEEDCAST) {
                if constexpr (!IsSame<T, float>::value) {
                    Cast(z_local_fp32[roundOffset], gamma_local, RoundMode::CAST_NONE, num_last_dim);
                    pipe_barrier(PIPE_V);
                    Mul(y_local_fp32, x_local_fp32, z_local_fp32[roundOffset], num_last_dim);
                    Cast(x_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                    pipe_barrier(PIPE_V);
                    Add(z_local_fp32[roundOffset], y_local_fp32, x_local_fp32, num_last_dim);
                    pipe_barrier(PIPE_V);
                    if constexpr (IsSame<T, half>::value) {
                        Cast(y_local[roundOffset], z_local_fp32[roundOffset], RoundMode::CAST_NONE, num_last_dim);
                    } else {
                        Cast(y_local[roundOffset], z_local_fp32[roundOffset], RoundMode::CAST_RINT, num_last_dim);
                    }
                    pipe_barrier(PIPE_V);
                } else {
                    Cast(z_local_fp32[roundOffset], gamma_local, RoundMode::CAST_NONE, num_last_dim);
                    pipe_barrier(PIPE_V);
                    Mul(y_local_fp32, x_local_fp32, z_local_fp32[roundOffset], num_last_dim);
                    Cast(x_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                    pipe_barrier(PIPE_V);
                    Add(y_local[roundOffset], y_local_fp32, x_local_fp32, num_last_dim);
                    pipe_barrier(PIPE_V);
                }
            } else {
                if constexpr (!IsSame<T, float>::value) {
                    Mul(y_local_fp32, x_local_fp32, gamma_local, num_last_dim);
                    pipe_barrier(PIPE_V);
                    Add(z_local_fp32[roundOffset], y_local_fp32, beta_local, num_last_dim);
                    pipe_barrier(PIPE_V);
                    if constexpr (IsSame<T, half>::value) {
                        Cast(y_local[roundOffset], z_local_fp32[roundOffset], RoundMode::CAST_NONE, num_last_dim);
                    } else {
                        Cast(y_local[roundOffset], z_local_fp32[roundOffset], RoundMode::CAST_RINT, num_last_dim);
                    }
                    pipe_barrier(PIPE_V);
                } else {
                    Mul(y_local_fp32, x_local_fp32, gamma_local, num_last_dim);
                    pipe_barrier(PIPE_V);
                    Add(y_local[roundOffset], y_local_fp32, beta_local, num_last_dim);
                    pipe_barrier(PIPE_V);
                }
            }
        }
#if OUTPUT_MEAN_RSTD == 1
        mean_que.EnQue(mean_local);
        rstd_que.EnQue(rstd_local);
#endif
        y_que.EnQue(y_local);
    }

    __aicore__ inline void precision_compute_big_n(
        int32_t nums, LocalTensor<T_GAMMA> &gamma_local, LocalTensor<T_GAMMA> &beta_local)
    {
        LocalTensor<T> y_local = y_que.template AllocTensor<T>();
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
#if OUTPUT_MEAN_RSTD == 1
        auto mean_local = mean_que.template AllocTensor<float>();
        auto rstd_local = rstd_que.template AllocTensor<float>();
#else
        auto mean_local = x_local_fp32[nums * ONE_BLK_FLOAT_NUM];
        auto rstd_local = x_local_fp32[nums * ONE_BLK_FLOAT_NUM];
#endif
        int32_t elementNum = num_last_dim_aligned * nums;
        Muls(y_local_fp32, z_local_fp32, aveNum, elementNum);
        pipe_barrier(PIPE_V);
        ReduceSumShort(mean_local, y_local_fp32, x_local_fp32, num_last_dim_aligned, num_last_dim, nums);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (int32_t idx = nums - 1; idx >= 0; idx--) {
            uint32_t offset = idx * num_last_dim_aligned;
            float meanTmp = mean_local.GetValue(idx);
            Adds(z_local_fp32[offset], z_local_fp32[offset], meanTmp * (-1), num_last_dim);
        }
        pipe_barrier(PIPE_V);
        Mul(y_local_fp32, z_local_fp32, z_local_fp32, elementNum);
        pipe_barrier(PIPE_V);
        Muls(y_local_fp32, y_local_fp32, aveNum, elementNum);
        pipe_barrier(PIPE_V);
        ReduceSumShort(rstd_local, y_local_fp32, x_local_fp32, num_last_dim_aligned, num_last_dim, nums);
        pipe_barrier(PIPE_V);
        Adds(rstd_local, rstd_local, eps, nums);
        pipe_barrier(PIPE_V);
        Sqrt(rstd_local, rstd_local, nums);
        Duplicate(y_local_fp32, (float)1, nums);
        pipe_barrier(PIPE_V);
        Div(rstd_local, y_local_fp32, rstd_local, nums);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (int32_t idx = nums - 1; idx >= 0; idx--) {
            uint32_t offset = idx * num_last_dim_aligned;
            float rstdTmp = rstd_local.GetValue(idx);
            Muls(z_local_fp32[offset], z_local_fp32[offset], rstdTmp, num_last_dim);
        }
        pipe_barrier(PIPE_V);

        if constexpr (IS_BETAGAMMA_NEEDCAST) {
            if constexpr (!IsSame<T, float>::value) {
                Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
                Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                pipe_barrier(PIPE_V);
                for (auto i = 0; i < nums; i++) {
                    Mul(z_local_fp32[i * num_last_dim_aligned],
                        z_local_fp32[i * num_last_dim_aligned],
                        x_local_fp32,
                        num_last_dim);
                }
                pipe_barrier(PIPE_V);
                for (auto i = 0; i < nums; i++) {
                    Add(z_local_fp32[i * num_last_dim_aligned],
                        z_local_fp32[i * num_last_dim_aligned],
                        y_local_fp32,
                        num_last_dim);
                }
                pipe_barrier(PIPE_V);
                if constexpr (IsSame<T, half>::value) {
                    Cast(y_local, z_local_fp32, RoundMode::CAST_NONE, elementNum);
                } else {
                    Cast(y_local, z_local_fp32, RoundMode::CAST_RINT, elementNum);
                }
            } else {
                Cast(x_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
                Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                pipe_barrier(PIPE_V);
                for (auto i = 0; i < nums; i++) {
                    Mul(z_local_fp32[i * num_last_dim_aligned],
                        z_local_fp32[i * num_last_dim_aligned],
                        x_local_fp32,
                        num_last_dim);
                }
                pipe_barrier(PIPE_V);
                for (auto i = 0; i < nums; i++) {
                    Add(y_local[i * num_last_dim_aligned],
                        z_local_fp32[i * num_last_dim_aligned],
                        y_local_fp32,
                        num_last_dim);
                }
                pipe_barrier(PIPE_V);
            }
        } else {
            if constexpr (!IsSame<T, float>::value) {
                for (auto i = 0; i < nums; i++) {
                    Mul(z_local_fp32[i * num_last_dim_aligned],
                        z_local_fp32[i * num_last_dim_aligned],
                        gamma_local,
                        num_last_dim);
                }
                pipe_barrier(PIPE_V);
                for (auto i = 0; i < nums; i++) {
                    Add(z_local_fp32[i * num_last_dim_aligned],
                        z_local_fp32[i * num_last_dim_aligned],
                        beta_local,
                        num_last_dim);
                }
                pipe_barrier(PIPE_V);
                if constexpr (IsSame<T, half>::value) {
                    Cast(y_local, z_local_fp32, RoundMode::CAST_NONE, elementNum);
                } else {
                    Cast(y_local, z_local_fp32, RoundMode::CAST_RINT, elementNum);
                }
            } else {
                for (auto i = 0; i < nums; i++) {
                    Mul(y_local[i * num_last_dim_aligned],
                        z_local_fp32[i * num_last_dim_aligned],
                        gamma_local,
                        num_last_dim);
                }
                pipe_barrier(PIPE_V);
                for (auto i = 0; i < nums; i++) {
                    Add(y_local[i * num_last_dim_aligned], y_local[i * num_last_dim_aligned], beta_local, num_last_dim);
                }
                pipe_barrier(PIPE_V);
            }
        }

#if OUTPUT_MEAN_RSTD == 1
        mean_que.EnQue(mean_local);
        rstd_que.EnQue(rstd_local);
#endif
        y_que.EnQue(y_local);
    }

    __aicore__ inline void precision_compute_single_row(
        LocalTensor<T_GAMMA> &gamma_local, LocalTensor<T_GAMMA> &beta_local)
    {
#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> mean_local = mean_que.template AllocTensor<float>();
        LocalTensor<float> rstd_local = rstd_que.template AllocTensor<float>();
#endif
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        Muls(y_local_fp32, x_local_fp32, aveNum, num_last_dim);
        pipe_barrier(PIPE_V);
        float ave_local_temp = ReduceSumFP32(y_local_fp32, num_last_dim);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        Adds(x_local_fp32, x_local_fp32, ave_local_temp * -1, num_last_dim);
        pipe_barrier(PIPE_V);
        Mul(y_local_fp32, x_local_fp32, x_local_fp32, num_last_dim);
        pipe_barrier(PIPE_V);
        Muls(y_local_fp32, y_local_fp32, aveNum, num_last_dim);
        pipe_barrier(PIPE_V);
        float var_local_temp = ReduceSumFP32(y_local_fp32, num_last_dim);
        float rstd_local_temp = 1 / sqrt(var_local_temp + eps);
#if OUTPUT_MEAN_RSTD == 1
        mean_local.SetValue(0, ave_local_temp);
        rstd_local.SetValue(0, rstd_local_temp);
#endif
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        Muls(x_local_fp32, x_local_fp32, rstd_local_temp, num_last_dim);
        pipe_barrier(PIPE_V);
        LocalTensor<T> y_local = y_que.template AllocTensor<T>();

        if constexpr (IS_BETAGAMMA_NEEDCAST) {
            if constexpr (!IsSame<T, float>::value) {
                Cast(y_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
                pipe_barrier(PIPE_V);
                Mul(x_local_fp32, x_local_fp32, y_local_fp32, num_last_dim);
                Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                pipe_barrier(PIPE_V);
                Add(x_local_fp32, x_local_fp32, y_local_fp32, num_last_dim);
                pipe_barrier(PIPE_V);
                if constexpr (IsSame<T, half>::value) {
                    Cast(y_local, x_local_fp32, RoundMode::CAST_NONE, num_last_dim);
                } else {
                    Cast(y_local, x_local_fp32, RoundMode::CAST_RINT, num_last_dim);
                }
            } else {
                Cast(y_local_fp32, gamma_local, RoundMode::CAST_NONE, num_last_dim);
                pipe_barrier(PIPE_V);
                Mul(x_local_fp32, x_local_fp32, y_local_fp32, num_last_dim);
                Cast(y_local_fp32, beta_local, RoundMode::CAST_NONE, num_last_dim);
                pipe_barrier(PIPE_V);
                Add(y_local, x_local_fp32, y_local_fp32, num_last_dim);
                pipe_barrier(PIPE_V);
            }
        } else {
            if constexpr (!IsSame<T, float>::value) {
                Mul(x_local_fp32, x_local_fp32, gamma_local, num_last_dim);
                pipe_barrier(PIPE_V);
                Add(x_local_fp32, x_local_fp32, beta_local, num_last_dim);
                pipe_barrier(PIPE_V);
                if constexpr (IsSame<T, half>::value) {
                    Cast(y_local, x_local_fp32, RoundMode::CAST_NONE, num_last_dim);
                } else {
                    Cast(y_local, x_local_fp32, RoundMode::CAST_RINT, num_last_dim);
                }
            } else {
                Mul(y_local_fp32, x_local_fp32, gamma_local, num_last_dim);
                pipe_barrier(PIPE_V);
                Add(y_local, y_local_fp32, beta_local, num_last_dim);
            }
        }

        y_que.EnQue(y_local);
#if OUTPUT_MEAN_RSTD == 1
        mean_que.EnQue(mean_local);
        rstd_que.EnQue(rstd_local);
#endif
    }

    __aicore__ inline void CopyOut(int32_t row_idx, int32_t row_count)
    {
        LocalTensor<T> res = y_que.template DeQue<T>();
        uint32_t gm_offset = row_idx * row_step * num_last_dim;
        DataCopyEx(y_gm[gm_offset], res, num_last_dim, row_count);
        y_que.FreeTensor(res);

#if OUTPUT_MEAN_RSTD == 1
        uint32_t gm_offset_mean = row_idx * row_step;
        LocalTensor<float> mean = mean_que.template DeQue<float>();
        LocalTensor<float> rstd = rstd_que.template DeQue<float>();
        DataCopyEx(mean_gm[gm_offset_mean], mean, row_count);
        DataCopyEx(rstd_gm[gm_offset_mean], rstd, row_count);
        mean_que.FreeTensor(mean);
        rstd_que.FreeTensor(rstd);
#endif
    }

    __aicore__ inline void CopyOutSlicePhase0(int32_t row_idx, int32_t size, int32_t offset = 0)
    {
        LocalTensor<T> x = x_que.template DeQue<T>();
        uint32_t gm_offset = row_idx * row_step * num_last_dim + offset;
        DataCopyEx(x_gm[gm_offset], x, size);
        x_que.FreeTensor(x);
    }

    __aicore__ inline void CopyOutSlicePhase1(int32_t row_idx, int32_t size, int32_t offset = 0)
    {
        LocalTensor<T> res = y_que.template DeQue<T>();
        uint32_t gm_offset = row_idx * row_step * num_last_dim + offset;
        DataCopyEx(y_gm[gm_offset], res, size);
        y_que.FreeTensor(res);
    }

    __aicore__ inline void CopyOutSlicePhase2(int32_t row_idx)
    {
#if OUTPUT_MEAN_RSTD == 1
        LocalTensor<float> mean = mean_que.template DeQue<float>();
        LocalTensor<float> rstd = rstd_que.template DeQue<float>();
        DataCopyEx(mean_gm[row_idx * row_step], mean, 1);
        DataCopyEx(rstd_gm[row_idx * row_step], rstd, 1);
        mean_que.FreeTensor(mean);
        rstd_que.FreeTensor(rstd);
#endif
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> gamma_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> beta_que;
    TQue<QuePosition::VECIN, BUFFER_NUM> bias_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> x_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> y_que;
#if OUTPUT_MEAN_RSTD == 1
    TQue<QuePosition::VECOUT, BUFFER_NUM> mean_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rstd_que;
#endif
    TBuf<TPosition::VECCALC> x_buf_fp32;
    TBuf<TPosition::VECCALC> y_buf_fp32;
    TBuf<TPosition::VECCALC> z_buf_fp32;
    GlobalTensor<T_X1> x1_gm;
    GlobalTensor<T_X2> x2_gm;
    GlobalTensor<T_GAMMA> gamma_gm;
    GlobalTensor<T_GAMMA> beta_gm;
    GlobalTensor<T> bias_gm;
    GlobalTensor<T> y_gm;
    GlobalTensor<T> x_gm;
    GlobalTensor<float> mean_gm;
    GlobalTensor<float> rstd_gm;
    GlobalTensor<float> workspace_gm;
    uint32_t num_core;
    uint32_t num_first_dim;
    uint32_t num_last_dim;
    uint32_t row_step;
    uint32_t row_work;
    uint32_t gm_offset_;
    uint32_t row_tail_;
    uint32_t col_tail;
    uint32_t col_move_cnt;
    uint32_t first_dim_per_time;
    uint32_t last_dim_per_time;
    uint32_t nl_first_dim_per_core;
    uint32_t l_first_dim_per_core;
    float eps;
    float aveNum;
    bool lastDimPad = false;
    size_t num_last_dim_aligned;
    bool lastDimPadMixDtype = false;
    size_t numLastDimAlignedMixDtype;
};

// 核函数入口
extern "C" __global__ __aicore__ void add_layer_norm(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias,
    GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling) {
    // 参数说明：
    //   x1 (aclTensor*，输入)：AddLayerNorm中加法计算的输入，参与 x1 + x2 + bias 计算并进行层归一化，Device 侧的 aclTensor，
    //                          支持 1-8 维，数据格式为 ND，数据类型支持 FLOAT32/FLOAT16 (昇腾310P)，另加 BFLOAT16 (昇腾910B/910_93)。
    //   x2 (aclTensor*，输入)：AddLayerNorm中加法计算的输入，参与 x1 + x2 + bias 计算，Device 侧的 aclTensor，shape 与 x1 一致，
    //                          数据格式为 ND，数据类型支持 FLOAT32/FLOAT16 (昇腾310P)，另加 BFLOAT16 (昇腾910B/910_93)。
    //   gamma (aclTensor*，输入)：LayerNorm中的 gamma 参数，Device 侧的 aclTensor，支持 1-8 维，与 x1 需要归一化的维度值相同，
    //                             数据格式为 ND，数据类型支持 FLOAT32/FLOAT16 (昇腾310P)，另加 BFLOAT16 (昇腾910B/910_93)。
    //   beta (aclTensor*，输入)：LayerNorm中的 beta 参数，Device 侧的 aclTensor，支持 1-8 维，与 x1 需要归一化的维度值相同，
    //                            数据格式为 ND，数据类型支持 FLOAT32/FLOAT16 (昇腾310P)，另加 BFLOAT16 (昇腾910B/910_93)。
    //   bias (aclTensor*，输入)：可选输入，加法计算的偏置，参与 x1 + x2 + bias 计算，Device 侧的 aclTensor，shape 可与 gamma/beta 或 x1/x2 一致，
    //                            支持 1-8 维，数据格式为 ND，数据类型支持 FLOAT32/FLOAT16 (昇腾310P)，另加 BFLOAT16 (昇腾910B/910_93)。
    //   y (aclTensor*，输出)：LayerNorm计算结果输出，Device 侧的 aclTensor，shape 与 x1/x2 一致，数据格式为 ND，
    //                         数据类型支持 FLOAT32/FLOAT16 (昇腾310P)，另加 BFLOAT16 (昇腾910B/910_93)。
    //   mean (aclTensor*，输出)：LayerNorm中 (x1 + x2 + bias) 的均值，Device 侧的 aclTensor，数据类型为 FLOAT32，shape 与 x1 满足 broadcast 关系，
    //                           数据格式为 ND，无效于昇腾310P。计算逻辑：mean = np.mean(x1 + x2 + bias)。
    //   rstd (aclTensor*，输出)：LayerNorm中 rstd (1/sqrt(var + epsilon)) 结果，Device 侧的 aclTensor，数据类型为 FLOAT32，shape 与 x1 满足 broadcast 关系，
    //                           数据格式为 ND，无效于昇腾310P。计算逻辑：rstd = np.power((np.var(x1 + x2 + bias) + epsilon), (-0.5))。
    //   x (aclTensor*，输出)：可选输出 (x1 + x2 + bias) 结果，Device 侧的 aclTensor，shape 与 x1/x2 一致，数据格式为 ND，
    //                         数据类型支持 FLOAT32/FLOAT16 (昇腾310P)，另加 BFLOAT16 (昇腾910B/910_93)。
    //   workspace：Device 侧的工作空间地址。
    //   tiling：tiling 结构体在 Device 侧的首地址。
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);

#define INIT_AND_PROCESS                 \
    op.Init(x1,                          \
        x2,                              \
        gamma,                           \
        beta,                            \
        bias,                            \
        y,                               \
        mean,                            \
        rstd,                            \
        x,                               \
        workspace,                       \
        tiling_data.numCore,             \
        tiling_data.numLastDim,          \
        tiling_data.numFirstDim,         \
        tiling_data.firstDimPerCore,     \
        tiling_data.firstDimPerCoreTail, \
        tiling_data.firstDimPerTime,     \
        tiling_data.lastDimPerTime,      \
        tiling_data.eps,                 \
        tiling_data.aveFactor,           \
        tiling_data.colMoveCnt,          \
        tiling_data.colTail,             \
        tiling_data.workspaceSize);      \
    op.Process()
    if (TILING_KEY_IS(0)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 0> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(10)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 10> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(20)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 20> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(30)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 30> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(40)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 40> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(50)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 50> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(1)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 1> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(11)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 11> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(21)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 21> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(31)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 31> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(41)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 41> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(51)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 51> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 2> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(12)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 12> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(22)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 22> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(32)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 32> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(42)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 42> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(52)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 52> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(100)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 100> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(110)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 110> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(120)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 120> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(130)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 130> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(140)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 140> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(150)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 150> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(101)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 101> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(111)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 111> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(121)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 121> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(131)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 131> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(141)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 141> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(151)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 151> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(102)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 102> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(112)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 112> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(122)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 122> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(132)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 132> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(142)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 142> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(152)) {
        KernelAddLayerNorm<DTYPE_X1, DTYPE_X2, DTYPE_GAMMA, DTYPE_Y, 152> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(62)) {  // Better UB begin
        KernelAddLayerNormBetterUB<half, half, half, half, 62> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(162)) {
        KernelAddLayerNormBetterUB<half, half, half, half, 162> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(70)) {  // Normal Special Reduce begin
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 70> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(170)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 170> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(80)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 80> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(180)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 180> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(72)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 72> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(172)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 172> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(82)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 82> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(182)) {
        KernelAddLayerNormNormalSpecialReduce<half, half, half, half, 182> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(190)) {  // Single Row Less Tensor begin
        KernelAddLayerNormSingleRowLessTensor<DTYPE_X1, DTYPE_X2, float, DTYPE_Y, 190> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(192)) {
        KernelAddLayerNormSingleRowLessTensor<DTYPE_X1, DTYPE_X2, float, DTYPE_Y, 192> op(&pipe);
        INIT_AND_PROCESS;
    }
}
