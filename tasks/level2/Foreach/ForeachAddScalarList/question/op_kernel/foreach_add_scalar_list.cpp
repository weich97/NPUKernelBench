/*!
 * \file foreach_add_scalar_list.cpp
 * \brief
 */

#include "kernel_operator.h"

// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_one_scalar_list_binary.h"
 
 using namespace AscendC;
 using namespace Common::OpKernel;
 
extern "C" __global__ __aicore__ void foreach_add_scalar_list(GM_ADDR inputs, GM_ADDR scalar,
    GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
}
 