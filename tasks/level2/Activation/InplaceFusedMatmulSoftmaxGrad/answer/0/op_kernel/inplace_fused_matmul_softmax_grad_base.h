#ifndef INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_BASE_H
#define INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace InplaceFusedMatmulSoftmaxGradOpt {
using namespace AscendC;
using namespace matmul;

template <typename cubeDataType, const MatmulConfig &MM_CFG = CFG_NORM>
struct MMType {
    using AT = MatmulType<TPosition::GM, CubeFormat::ND, cubeDataType>;
    using BT = MatmulType<TPosition::GM, CubeFormat::ND, cubeDataType, true>;
    using CT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, cubeDataType>;
    using MT = matmul::Matmul<AT, BT, CT, BiasT, MM_CFG>;
};

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t ELE_NUM_FP32 = 8;

template <typename mmType, typename dataType, bool isAligned = false, bool isCast = false>
class InplaceFusedMatmulSoftmaxGradBase {
public:
    __aicore__ inline InplaceFusedMatmulSoftmaxGradBase(mmType &mm) : mm1_(mm)
    {
    }

    template <typename T>
    __aicore__ inline T CeilDiv(T x, T y)
    {
        return y == 0 ? 0 : (x + y - 1) / y;
    }

    __aicore__ inline float GetMax(float a, float b)
    {
        return a > b ? a : b;
    }

    template <typename T>
    __aicore__ inline T AlignUp(T num, T div)
    {
        return (div == 0) ? 0 : (num + div - 1) / div * div;
    }

public:
    TPipe *pPipe_ = nullptr;
    mmType &mm1_;
    /* tiling data */
    BaseTiling baseTilingData_;
    SoftMaxTiling headSoftMaxGradTilingData_;
    TCubeTiling cubeTilingData_;
    
    /* variable */
    AscendC::GlobalTensor<dataType> softmaxOutputGm_; // Implementation note.
    AscendC::GlobalTensor<dataType> gradOutputGm_; // Implementation note.
    AscendC::GlobalTensor<dataType> valuesGm_; // Implementation note.
    AscendC::GlobalTensor<float> gradSoftmaxGm_; // Implementation note.
    AscendC::GlobalTensor<dataType> gradXGm_; // Implementation note.
    AscendC::LocalTensor<float> softmaxTmpUb1_; // Implementation note.
    AscendC::LocalTensor<float> softmaxTmpUb2_; // Implementation note.
    AscendC::LocalTensor<float> softmaxOutputUB32Temp; // Implementation note.
    AscendC::LocalTensor<float> outGradX32Temp; // Implementation note.

    uint32_t blockIdx_{0};
    uint32_t rowLen_{0};

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSoftmaxOutput_; // Implementation note.
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGradSoftmax_; // Implementation note.
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGradX_; // Implementation note.
    TBuf<TPosition::VECCALC> SoftmaxOutput32Temp_; // Implementation note.
    TBuf<TPosition::VECCALC> outGradX32Temp_; // Implementation note.
    TBuf<TPosition::VECCALC> sharedTempBuf1_; // Implementation note.
    TBuf<TPosition::VECCALC> sharedTempBuf2_; // Implementation note.
};
}  // namespace InplaceFusedMatmulSoftmaxGradOpt
#endif  // INPLACE_FUSED_MATMUL_SOFTMAX_GRAD_BASE_H