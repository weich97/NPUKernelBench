add_ops_compile_options(
        OP_NAME CosMath
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
op_host/cos_math.cpp
)

target_sources(optiling PRIVATE
        op_host/cos_math.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/cos_math.cpp
)

install(FILES op_kernel/cos_math.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)