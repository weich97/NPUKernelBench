#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void flash_attention_score_with_large_head_dim(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR attention_out, GM_ADDR workspace, GM_ADDR tiling) {
}