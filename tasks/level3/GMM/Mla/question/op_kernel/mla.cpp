#include <kernel_operator.h>
using namespace AscendC;

extern "C" __global__ __aicore__ void mla(GM_ADDR query_nope, GM_ADDR query_rope, GM_ADDR kv_nope_cache,
                                            GM_ADDR kv_rope_cache, GM_ADDR block_tables, GM_ADDR out,
                                            GM_ADDR workspace, GM_ADDR tiling)
{
}