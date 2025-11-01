#include "kernel_operator.h"

using namespace AscendC;


// 核函数入口
extern "C" __global__ __aicore__ void scatter_list(GM_ADDR var, GM_ADDR indice, GM_ADDR updates, GM_ADDR mask,
                                                   GM_ADDR varOut, GM_ADDR workspace, GM_ADDR tiling)
{
}