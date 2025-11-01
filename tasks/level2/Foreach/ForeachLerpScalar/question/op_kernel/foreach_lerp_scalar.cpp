/*!
 * \file foreach_acos.cpp
 * \brief
 */

#include "kernel_operator.h"
// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_one_scalar_binary.h"

using namespace AscendC;
using namespace Common::OpKernel;

extern "C" __global__ __aicore__ void foreach_lerp_scalar(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
}
