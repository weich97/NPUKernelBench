# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (TARGET json_build)
    return()
endif ()

include(ExternalProject)

if (CANN_PKG)
    set(REQ_URL "${CANN_PKG}/include/JSON-for-Modern-CPP-v3.11.2.zip")
elseif (ENABLE_GITHUB)
    set(REQ_URL "https://github.com/nlohmann/json/releases/download/v3.11.2/include.zip")
else ()
    set(REQ_URL "https://gitee.com/mirrors/JSON-for-Modern-CPP/repository/archive/v3.11.2.zip")
endif ()

ExternalProject_Add(json_build
                    URL ${REQ_URL}
                    CONFIGURE_COMMAND ${CMAKE_COMMAND}
                        -DJSON_MultipleHeaders=ON
                        -DJSON_BuildTests=OFF
                        -DBUILD_SHARED_LIBS=OFF
                        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/json
                        <SOURCE_DIR>
                    EXCLUDE_FROM_ALL TRUE
)
