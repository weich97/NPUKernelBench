#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void non_finite_check_op(GM_ADDR tensor_list, GM_ADDR found_flag, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
   
}