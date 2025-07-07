cmake_minimum_required(VERSION 3.27)

set(PROJECT_NAME SmollNet)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(fmt REQUIRED)
find_package(pybind11 REQUIRED)

add_library(${PROJECT_NAME}
    src/tensor.cpp
    src/neuralnet.cpp
    src/autograd.cpp
    src/sgd.cpp
    src/layer_norm.cpp
)

set(CPP_LIBNAME SmollNet)

set(CUDA_LIBNAME SmollNetCUDA)
include(cmake/cuda.cmake)

target_include_directories(${CPP_LIBNAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(${CPP_LIBNAME}
  PUBLIC
    ${CUDA_LIBNAME}
    fmt::fmt

  PRIVATE
    pybind11::headers
    pybind11::module
)

target_compile_features(${CPP_LIBNAME} PUBLIC cxx_std_23)
set_target_properties(${CPP_LIBNAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(${CPP_LIBNAME}
  PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g>
)


include(cmake/install.cmake)
include(cmake/bindings.cmake)
