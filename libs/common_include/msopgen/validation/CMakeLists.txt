cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(python_extension)
message("Using python: ${Python3_EXECUTABLE}")
find_package(Python3 COMPONENTS Interpreter Development)
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}")
message(${CMAKE_PREFIX_PATH})
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(torch_act PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_SITELIB}/torch_npu/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${SHARED_LIB_DIR}
    ${SHARED_LIB_DIR}/autogen
    ${ASCEND_HOME_PATH}/include/experiment/msprof
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include)

link_directories(${ASCEND_HOME_PATH}/lib64
    ${CMAKE_BINARY_DIR}
    ${Python3_SITELIB}/torch_npu/lib
    ${SHARED_LIB_DIR})

link_libraries(cust_opapi "${TORCH_LIBRARIES}" torch_npu tiling_api ascendcl acl_op_compiler nnopbase)

pybind11_add_module(kernel_gen_ops SHARED pybind.cpp)

install(TARGETS kernel_gen_ops DESTINATION .)