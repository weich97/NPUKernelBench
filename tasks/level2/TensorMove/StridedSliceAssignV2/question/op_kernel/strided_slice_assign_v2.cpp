#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void strided_slice_assign_v2(GM_ADDR var, GM_ADDR input_value, GM_ADDR begin,
                                                              GM_ADDR end, GM_ADDR strides, GM_ADDR axes,
                                                              GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling)
{
}