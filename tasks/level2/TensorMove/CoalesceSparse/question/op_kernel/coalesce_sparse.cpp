#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void coalesce_sparse(GM_ADDR unique_len, GM_ADDR unique_indices, GM_ADDR indices,
                                                      GM_ADDR values, GM_ADDR new_indices, GM_ADDR new_value,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{

}