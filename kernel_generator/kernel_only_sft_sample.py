KERNEL_ONLY_EXAMPLE_CODE = '''
【参考案例】
以下是一个类似算子实现的案例：

这是一个样例Ascend C代码（仅供参考格式和语法）。接下来的内容是对该用例的任务描述，用来精确说明要写一个什么样的算子。案例任务描述如下所示：
【输入文件】
下面是第一个文件："api_desc.md"，是待实现算子的接口描述和公式定义相关内容：
# AddCustom

## 支持的数据类型
- torch.float16

## 功能描述

### 算子功能
实现了两个数据相加，返回相加结果的功能。

### 计算公式

  $$
  z = x + y
  $$

## 约束与限制
- 输入输出张量数据格式支持ND。

下面是第二个文件："test_cases.csv"，其设定了待实现算子需要重点考虑的典型shape以及DataType信息，用于设计最优tiling策略：
case_id            0
shape      [48, 1024]
dtype        float16
Name: 0, dtype: object


模型最终输出的结果应该是：

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


这个例子展示了如何设计tiling结构体和tiling策略，以及如何正确实现一个kernel。请参考这种结构来组织你的代码。
'''

KERNEL_ONLY_CODE_REQUIREMENT = '''
1. 不要使用GET_TILING_DATA来获取tiling，而是参考示例的TilingDataDef写法来获取tiling
2. 代码实现能满足当前test_cases.csv要求的输入信息即可，api_desc.md描述中与当前输入信息无关的功能可以不实现
3. 注意参考硬件规格信息来进行分块及搬运设计，不要出现内部地址越界等问题
4. 注意语法严谨和正确，生成的过程中反复检查，不要出现任何未定义的变量和类，保证代码可执行和功能正确
5. 注意代码中不要使用DTYPE_X 和 DTYPE_Y等包含DTYPE的命名来指代数据类型
6. 注意Ascend C官方定义的bfloat16类型名是bfloat16_t，而不是bfloat16
7. 注意Ascend C官方定义的float16类型名是float16_t，而不是float16
8. 注意host侧调用kernel侧时，使用了全部的aic或aiv核，其中aic有24个核，aiv有48个核
9. 使用DataCopy接口进行数据搬运，搬运的数据长度和操作数的起始地址（UB上）必须保证32字节对齐 
'''