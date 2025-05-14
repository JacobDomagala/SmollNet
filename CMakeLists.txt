cmake_minimum_required(VERSION 3.23)

set(PROJECT_NAME SmollNet)

# CUDA setup
set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "" FORCE)
set(CUDAToolkit_ROOT   "${CUDA_ROOT}"           CACHE PATH     "" FORCE)
set(CMAKE_CUDA_HOST_COMPILER ${CXX_COMPILER}    CACHE FILEPATH "" FORCE)

set(CMAKE_CC_COMPILER ${CC_COMPILER})
set(CMAKE_CXX_COMPILER ${CXX_COMPILER})

project(${PROJECT_NAME} LANGUAGES CXX CUDA)


add_library(${PROJECT_NAME} src/tensor.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE "${CUDAToolkit_ROOT}/include" src/)
