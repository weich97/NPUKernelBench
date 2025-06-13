/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "aoe/op_tuning_tiling/conv2d_dw_tuning_tiling.h"

namespace tuningtiling {
DECLARE_STRUCT_RELATE_WITH_OP_V2(Conv2DBackpropFilter, Conv2DDwInputArgs, a_shape_n, a_shape_h, a_shape_w, b_shape_h,
                              b_shape_w, c_shape_n, c_shape_c, c_shape_h, c_shape_w, groups, stride_h, stride_w,
                              dilation_h, dilation_w, pad_u, pad_d, pad_l, pad_r, a_dtype, b_dtype, c_dtype,
                              binary_mode, hf32_flag, reserved_params1, reserved_params2, reserved_params3,
                              reserved_params4, reserved_params5);
REGISTER_TUNING_TILING_CLASS(Conv2DBackpropFilter, Conv2DDwTunnerTiling);
}  // namespace tuningtiling