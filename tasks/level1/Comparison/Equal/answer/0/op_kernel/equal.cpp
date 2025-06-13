#include <type_traits>
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr float POSITIVE_ONE_FP32 = 1.0F;
constexpr int32_t POSITIVE_ONE_I32 = 1;
constexpr float MIN_ACCURACY_FP16 = 0.00000005960464477539063F;
constexpr float MAX_MUL_FP16 = 4096;
constexpr float MIN_ACCURACY_FP32 = 1.1754943508222875e-38;
constexpr float MAX_MUL_1_FP32 = 1125899906842624;
constexpr float MAX_MUL_2_FP32 = 67108864;
constexpr uint32_t BLOCK_SIZE = 32;

class KernelEqual
{
public:
  __aicore__ inline KernelEqual() {}
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                              uint32_t total_length, uint32_t tile_num_mean,
                              uint32_t tile_num_end, uint32_t tile_length_mean,
                              uint32_t tile_length_end, uint32_t block_length_mean,
                              uint32_t block_length_end)
  {
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    ResovleTiling(total_length, tile_num_mean, tile_num_end, tile_length_mean,
                  tile_length_end, block_length_mean, block_length_end);
    x1_gm.SetGlobalBuffer(
        (__gm__ DTYPE_X1 *)x1 + this->block_offset * GetBlockIdx(),
        this->block_length);
    x2_gm.SetGlobalBuffer(
        (__gm__ DTYPE_X1 *)x2 + this->block_offset * GetBlockIdx(),
        this->block_length);
    y_gm.SetGlobalBuffer((__gm__ int8_t *)y + this->block_offset * GetBlockIdx(),
                         this->block_length);

    pipe.InitBuffer(x1_inque, BUFFER_NUM, this->tile_cache * sizeof(DTYPE_X1));
    pipe.InitBuffer(x2_inque, BUFFER_NUM, this->tile_cache * sizeof(DTYPE_X1));
    pipe.InitBuffer(y_outque, BUFFER_NUM,
                    this->tile_cache * sizeof(int8_t) < BLOCK_SIZE
                        ? BLOCK_SIZE
                        : this->tile_cache * sizeof(int8_t));
    pipe.InitBuffer(calc_buf_1, this->tile_cache * sizeof(DTYPE_X1));
    pipe.InitBuffer(calc_buf_2, this->tile_cache * sizeof(half) < BLOCK_SIZE
                                    ? BLOCK_SIZE
                                    : this->tile_cache * sizeof(half));
    pipe.InitBuffer(calc_buf_3, this->tile_cache * sizeof(half) < BLOCK_SIZE
                                    ? BLOCK_SIZE
                                    : this->tile_cache * sizeof(half));
    pipe.InitBuffer(calc_buf_4, this->tile_cache * sizeof(float) < BLOCK_SIZE
                                    ? BLOCK_SIZE
                                    : this->tile_cache * sizeof(float));
  }
  __aicore__ inline void Process()
  {
    if (this->total_length <= BLOCK_SIZE / sizeof(DTYPE_X1))
    {
      CopyInPad(0);
      Compute(0);
      CopyOutPad(0);
      return;
    }
    int32_t loopCount = this->tile_num;
    for (int32_t i = 0; i < loopCount - 1; i++)
    {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
    if (GetBlockIdx() == (GetBlockNum() - 1))
    {
      CopyInPad(loopCount - 1);
      Compute(loopCount - 1);
      CopyOutPad(loopCount - 1);
    }
    else
    {
      CopyIn(loopCount - 1);
      Compute(loopCount - 1);
      CopyOut(loopCount - 1);
    }
  }

private:
  __aicore__ inline void ResovleTiling(
      uint32_t total_length, uint32_t tile_num_mean, uint32_t tile_num_end,
      uint32_t tile_length_mean, uint32_t tile_length_end, uint32_t block_length_mean,
      uint32_t block_length_end)
  {
    uint32_t pad32 = BLOCK_SIZE;
    this->total_length = total_length;
    if (GetBlockNum() >= 1 && GetBlockIdx() == (GetBlockNum() - 1))
    {
      this->block_length = block_length_end;
      this->tile_num = tile_num_end;
    }
    else
    {
      this->block_length = block_length_mean;
      this->tile_num = tile_num_mean;
    }
    this->block_offset = block_length_mean;
    this->tile_length = tile_length_mean;
    this->tile_cache = tile_length_mean;
    this->tile_length_end = tile_length_end;
    if (total_length < pad32)
    {
      this->block_offset = 0;
      this->tile_cache = pad32;
      this->block_length = pad32;
    }
  }
  __aicore__ inline void CopyIn(int32_t progress)
  {
    LocalTensor<DTYPE_X1> x1_local = x1_inque.AllocTensor<DTYPE_X1>();
    LocalTensor<DTYPE_X1> x2_local = x2_inque.AllocTensor<DTYPE_X1>();
    DataCopy(x1_local, x1_gm[progress * this->tile_cache], this->tile_cache);
    DataCopy(x2_local, x2_gm[progress * this->tile_cache], this->tile_cache);
    x1_inque.EnQue(x1_local);
    x2_inque.EnQue(x2_local);
  }
  __aicore__ inline void CopyInPad(int32_t progress)
  {
    LocalTensor<DTYPE_X1> x1_local = x1_inque.AllocTensor<DTYPE_X1>();
    LocalTensor<DTYPE_X1> x2_local = x2_inque.AllocTensor<DTYPE_X1>();
    DataCopy(x1_local, x1_gm[progress * this->tile_cache],
                      ((this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE));
    DataCopy(x2_local, x2_gm[progress * this->tile_cache],
                      ((this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE));
    x1_inque.EnQue(x1_local);
    x2_inque.EnQue(x2_local);
  }
  __aicore__ inline void Compute(int32_t progress)
  {
    if constexpr (std::is_same_v<DTYPE_X1, half>||std::is_same_v<DTYPE_X1, bfloat16_t>)
    {
      LocalTensor<half> x1_local = x1_inque.DeQue<half>();
      LocalTensor<half> x2_local = x2_inque.DeQue<half>();
      LocalTensor<int8_t> y_local = y_outque.AllocTensor<int8_t>();
      LocalTensor<half> y_compute = calc_buf_1.Get<half>();
      // Step 1: 计算差值 diff = x1 - x2
      Sub(y_compute, x1_local, x2_local, this->tile_cache);

      // Step 2: 取绝对值 abs_diff = |diff|
      Abs(y_compute, y_compute, this->tile_cache);

      // Step 3: 误差容差处理，将小于误差值的差值设置为 0
      Mins(y_compute, y_compute, (half)MIN_ACCURACY_FP16, this->tile_cache);

      // Step 4: 将所有非零值设置为 1
      Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache);
      Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache);

      // Step 5: 最终结果：将所有非零值设置为 1，零值保持为 0
      Duplicate(x1_local, (half)POSITIVE_ONE_FP32, this->tile_cache);
      Sub(y_compute, x1_local, y_compute, this->tile_cache);

      // 将结果转换为 FP16 类型并保存到输出张量
      Cast(y_local, y_compute, RoundMode::CAST_NONE, this->tile_cache);
      y_outque.EnQue<int8_t>(y_local);
      x1_inque.FreeTensor(x1_local);
      x2_inque.FreeTensor(x2_local);
    }
    else if constexpr (std::is_same_v<DTYPE_X1, float>)
    {
      LocalTensor<float> x1_local = x1_inque.DeQue<float>();
      LocalTensor<float> x2_local = x2_inque.DeQue<float>();
      LocalTensor<int8_t> y_local = y_outque.AllocTensor<int8_t>();
      LocalTensor<float> y_compute = calc_buf_1.Get<float>();

      LocalTensor<half> y_fp16 = calc_buf_2.Get<half>();

      // Step 1: 计算差值 diff = x1 - x2
      Sub(y_compute, x1_local, x2_local, this->tile_cache);

      // Step 2: 取绝对值 abs_diff = |diff|
      Abs(y_compute, y_compute, this->tile_cache);

      // Step 3: 误差容差处理，将小于误差值的差值设置为 0
      Mins(y_compute, y_compute, (float)MIN_ACCURACY_FP32, this->tile_cache);

      // Step 4: 将所有非零值设置为 1
      Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->tile_cache);
      Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->tile_cache);
      Muls(y_compute, y_compute, (float)MAX_MUL_2_FP32, this->tile_cache);

      // Step 5: 最终结果：将所有非零值设置为 1，零值保持为 0
      Duplicate(x1_local, (float)POSITIVE_ONE_FP32, this->tile_cache);
      Sub(y_compute, x1_local, y_compute, this->tile_cache);

      // 将结果转换为 FP16 类型并保存到输出张量
      Cast(y_fp16, y_compute, RoundMode::CAST_NONE, this->tile_cache);
      Cast(y_local, y_fp16, RoundMode::CAST_NONE, this->tile_cache);
      y_outque.EnQue<int8_t>(y_local);
      x1_inque.FreeTensor(x1_local);
      x2_inque.FreeTensor(x2_local);
    }
    else if constexpr (std::is_same_v<DTYPE_X1, int8_t>||std::is_same_v<DTYPE_X1, uint8_t>)
    {
      LocalTensor<int8_t> x1_local = x1_inque.DeQue<int8_t>();
      LocalTensor<int8_t> x2_local = x2_inque.DeQue<int8_t>();
      LocalTensor<int8_t> y_local = y_outque.AllocTensor<int8_t>();
      LocalTensor<int8_t> y_compute = calc_buf_1.Get<int8_t>();
      // Step 1: 初始化中间张量
      LocalTensor<half> x1_local_fp16 = calc_buf_2.Get<half>();
      LocalTensor<half> x2_local_fp16 = calc_buf_3.Get<half>();
      LocalTensor<half> y_local_fp16 = calc_buf_4.Get<half>();

      // Step 2: 将 int8 转换为 FP16
      Cast(x1_local_fp16, x1_local, RoundMode::CAST_NONE, this->tile_cache);
      Cast(x2_local_fp16, x2_local, RoundMode::CAST_NONE, this->tile_cache);

      // Step 3: 计算差值 diff = x1 - x2
      Sub(y_local_fp16, x1_local_fp16, x2_local_fp16, this->tile_cache);
      // Step 4: 取绝对值 abs_diff = |diff|
      Abs(y_local_fp16, y_local_fp16, this->tile_cache);

      // Step 5: 将所有非零值设置为 1
      Mins(y_local_fp16, y_local_fp16, (half)POSITIVE_ONE_FP32, this->tile_cache);

      // Step 6: 布尔值反转，将零值设为 1，非零值设为 0
      Duplicate(x1_local_fp16, (half)POSITIVE_ONE_FP32, this->tile_cache);
      Sub(y_local_fp16, x1_local_fp16, y_local_fp16, this->tile_cache);

      // Step 7: 将结果转换为 int8 并保存到输出张量
      Cast(y_local, y_local_fp16, RoundMode::CAST_NONE, this->tile_cache);
      y_outque.EnQue<int8_t>(y_local);
      x1_inque.FreeTensor(x1_local);
      x2_inque.FreeTensor(x2_local);
    }
    else if constexpr (std::is_same_v<DTYPE_X1, int32_t>||std::is_same_v<DTYPE_X1, uint32_t>)
    {
      LocalTensor<int32_t> x1_local = x1_inque.DeQue<int32_t>();
      LocalTensor<int32_t> x2_local = x2_inque.DeQue<int32_t>();
      LocalTensor<int8_t> y_local = y_outque.AllocTensor<int8_t>();
      LocalTensor<int32_t> y_compute = calc_buf_1.Get<int32_t>();

      LocalTensor<half> y_fp16 = calc_buf_3.Get<half>();
      LocalTensor<float> y_fp32 = calc_buf_4.Get<float>();

      Sub(y_compute, x1_local, x2_local, this->tile_cache);
      // y_compute->diff_fp32,int32->float
      Abs(y_compute.ReinterpretCast<float>(), y_compute.ReinterpretCast<float>(), this->tile_cache);

      Mins(y_compute, y_compute, (int32_t)POSITIVE_ONE_I32, this->tile_cache);
      Duplicate(x1_local, (int32_t)POSITIVE_ONE_I32, this->tile_cache);
      Sub(y_compute, x1_local, y_compute, this->tile_cache);

      // 将结果转换为 FP16 类型并保存到输出张量
      // int32->float
      Cast(y_fp32, y_compute, RoundMode::CAST_NONE, this->tile_cache);
      // float->half
      Cast(y_fp16, y_fp32, RoundMode::CAST_NONE, this->tile_cache);
      Cast(y_local, y_fp16, RoundMode::CAST_NONE, this->tile_cache);

      y_outque.EnQue<int8_t>(y_local);
      x1_inque.FreeTensor(x1_local);
      x2_inque.FreeTensor(x2_local);
    }
  }
  __aicore__ inline void CopyOut(int32_t progress)
  {
    LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
    DataCopy(y_gm[progress * this->tile_cache], y_local,
                      this->tile_cache);
    y_outque.FreeTensor(y_local);
  }
  __aicore__ inline void CopyOutPad(int32_t progress)
  {
    LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
    DataCopy(y_gm[progress * this->tile_cache], y_local,
                      (this->tile_length_end + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
    y_outque.FreeTensor(y_local);
  }

private:
  TPipe pipe;
  TBuf<TPosition::VECCALC> calc_buf_1, calc_buf_2, calc_buf_3, calc_buf_4;
  TQue<QuePosition::VECIN, BUFFER_NUM> x1_inque, x2_inque;
  TQue<QuePosition::VECOUT, BUFFER_NUM> y_outque;
  GlobalTensor<DTYPE_X1> x1_gm, x2_gm;
  GlobalTensor<int8_t> y_gm;
  uint32_t total_length, block_length, block_offset, tile_num;
  uint32_t tile_cache, tile_length, tile_length_end;
};

extern "C" __global__ __aicore__ void equal(GM_ADDR x1, GM_ADDR x2,
                                            GM_ADDR y, GM_ADDR workspace,
                                            GM_ADDR tiling)
{
  GET_TILING_DATA(tiling_data, tiling);
  KernelEqual op;
  op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.tileNumMean,
          tiling_data.tileNumEnd, tiling_data.tileLengthMean,
          tiling_data.tileLengthEnd, tiling_data.blockLengthMean,
          tiling_data.blockLengthEnd);
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void equal_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x1,
              uint8_t *x2, uint8_t *y, uint8_t *workspace,
              uint8_t *tiling)
{
  equal<<<blockDim, l2ctrl, stream>>>(x1, x2, y, workspace, tiling);
}
#endif