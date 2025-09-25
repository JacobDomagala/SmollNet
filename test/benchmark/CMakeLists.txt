add_executable(variance_benchmark
  variance_benchmark.cpp
)

target_link_libraries(variance_benchmark
  PRIVATE
    ${CPP_LIBNAME}
)
