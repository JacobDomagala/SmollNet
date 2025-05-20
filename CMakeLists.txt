cmake_minimum_required(VERSION 3.27)

set(PROJECT_NAME SmollNet)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)

add_library(${PROJECT_NAME}
    src/tensor.cpp
    src/kernels.cu
)

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart fmt::fmt)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23 cuda_std_20)

include(cmake/install.cmake)
