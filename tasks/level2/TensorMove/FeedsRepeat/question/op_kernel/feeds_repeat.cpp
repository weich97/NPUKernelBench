#include "kernel_operator.h"

using namespace AscendC;


// 核函数入口
extern "C" __global__ __aicore__ void feeds_repeat(GM_ADDR feeds, GM_ADDR feeds_repeat_times, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
}