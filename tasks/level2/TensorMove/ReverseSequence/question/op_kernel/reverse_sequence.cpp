#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void reverse_sequence(GM_ADDR x, GM_ADDR seq_lengths, GM_ADDR y, GM_ADDR workspace,
                                                       GM_ADDR tiling) {
}