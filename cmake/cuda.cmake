find_package(CUDAToolkit REQUIRED)

add_library(${CUDA_LIBNAME}
  STATIC
    src/kernels.cu
    src/sum.cu
    src/variance.cu
)

target_link_libraries(${CUDA_LIBNAME}
  PUBLIC
    CUDA::cudart
    fmt::fmt
)

target_include_directories(${CUDA_LIBNAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(${CUDA_LIBNAME}
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_features(${CUDA_LIBNAME} PUBLIC cxx_std_23 cuda_std_20)

target_compile_options(${CUDA_LIBNAME}
  PRIVATE
    -lineinfo
)
