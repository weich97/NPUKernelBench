KERNEL_ONLY_EXAMPLE_CODE = '''
[Reference Example]
The following is an example of a similar Ascend C operator implementation. It is provided only as a reference for format, structure, and syntax.

[Input File: api_desc.md]
# AddCustom

## Supported Data Types
- torch.float16

## Functional Description

### Operator Function
This operator adds two input tensors element-wise and returns the result.

### Formula

$$
z = x + y
$$

## Constraints and Limitations
- Input and output tensors use the ND data format.

[Input File: test_cases.csv]
The test case specifies the representative shape and data type that should guide the tiling strategy:

case_id            0
shape      [48, 1024]
dtype        float16
Name: 0, dtype: object

The final model output should follow this structure:

<kernel_impl>
```cpp
#include "kernel_operator.h"

using namespace AscendC;

typedef struct {
    uint32_t totalLength;
    uint32_t tileNum;
} TilingDataDef;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float16_t *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float16_t *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float16_t *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float16_t));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float16_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float16_t));
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
        AscendC::LocalTensor<float16_t> xLocal = inQueueX.AllocTensor<float16_t>();
        AscendC::LocalTensor<float16_t> yLocal = inQueueY.AllocTensor<float16_t>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float16_t> xLocal = inQueueX.DeQue<float16_t>();
        AscendC::LocalTensor<float16_t> yLocal = inQueueY.DeQue<float16_t>();
        AscendC::LocalTensor<float16_t> zLocal = outQueueZ.AllocTensor<float16_t>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<float16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float16_t> zLocal = outQueueZ.DeQue<float16_t>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float16_t> xGm;
    AscendC::GlobalTensor<float16_t> yGm;
    AscendC::GlobalTensor<float16_t> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace)
{
    TilingDataDef tiling_data = {
        .totalLength = 49152,
        .tileNum = 1,
    };
    KernelAdd op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
```
</kernel_impl>

This example demonstrates how to define a tiling structure, choose a simple tiling strategy, and implement a kernel with explicit data movement and computation stages. Use this structure as a reference when organizing your own code.
'''

KERNEL_ONLY_CODE_REQUIREMENT = '''
1. Do not use `GET_TILING_DATA` to obtain tiling information. Follow the reference example and define a local `TilingDataDef` structure instead.
2. The implementation only needs to satisfy the input information required by the current `test_cases.csv`; functionality described in `api_desc.md` that is irrelevant to the current inputs may be omitted.
3. Design tiling and data movement according to the hardware specification, and avoid out-of-bounds accesses or invalid internal addresses.
4. Keep the syntax rigorous and correct. Re-check the generated code to avoid undefined variables, undefined classes, or non-executable code paths.
5. Do not use placeholder names such as `DTYPE_X` or `DTYPE_Y` to represent concrete data types.
6. The official Ascend C type name for bfloat16 is `bfloat16_t`, not `bfloat16`.
7. The official Ascend C type name for float16 is `float16_t`, not `float16`.
8. The host side launches the kernel using all available AIC or AIV cores; the target platform has 24 AIC cores and 48 AIV cores.
9. When using `DataCopy`, both the transfer length and the operand start address in UB must satisfy 32-byte alignment.
'''
