#ifndef COALESCE_SPARSE_TILING_H
#define COALESCE_SPARSE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CoalesceSparseTilingData)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, m);
TILING_DATA_FIELD_DEF(uint64_t, valueSize);
TILING_DATA_FIELD_DEF(uint64_t, taskNum);
TILING_DATA_FIELD_DEF(uint64_t, taskTail);
TILING_DATA_FIELD_DEF(uint64_t, moveOneSize);
TILING_DATA_FIELD_DEF(uint64_t, taskRepeatTimes);
TILING_DATA_FIELD_DEF(uint64_t, taskRepeatTail);
TILING_DATA_FIELD_DEF(uint64_t, taskTailRepeatTimes);
TILING_DATA_FIELD_DEF(uint64_t, taskTailRepeatTail);
TILING_DATA_FIELD_DEF(uint64_t, moveValueTimes);
TILING_DATA_FIELD_DEF(uint64_t, moveValueLen);
TILING_DATA_FIELD_DEF(uint64_t, moveValueTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CoalesceSparse, CoalesceSparseTilingData)
} // namespace optiling
#endif // COALESCE_SPARSE_TILING_H