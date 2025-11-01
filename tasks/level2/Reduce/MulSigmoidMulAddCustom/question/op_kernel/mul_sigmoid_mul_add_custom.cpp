#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ 
void mul_sigmoid_mul_add_custom(GM_ADDR input, GM_ADDR mulScalar1, GM_ADDR mulScalar2, GM_ADDR addScalar3, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {

}
