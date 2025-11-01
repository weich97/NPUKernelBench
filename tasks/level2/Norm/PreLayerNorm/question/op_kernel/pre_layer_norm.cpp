#include "kernel_operator.h"
using namespace AscendC;


extern "C" __global__ __aicore__ void pre_layer_norm(
    GM_ADDR x, GM_ADDR y, GM_ADDR gamma, GM_ADDR beta, GM_ADDR res_out,
    GM_ADDR workspace, GM_ADDR tiling) {
  
}