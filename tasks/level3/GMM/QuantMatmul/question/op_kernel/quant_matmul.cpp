#include <kernel_operator.h>
using namespace AscendC;
extern "C" __global__ __aicore__ void quant_matmul(GM_ADDR a, GM_ADDR b, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR d, GM_ADDR workspace, GM_ADDR tiling)
{

}