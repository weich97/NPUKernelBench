#include "kernel_operator.h"

using namespace AscendC;
 
extern "C" __global__ __aicore__ void foreach_add_scalar_list(GM_ADDR inputs, GM_ADDR scalar,
    GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
}
 