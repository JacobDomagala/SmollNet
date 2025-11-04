add_executable(welford_benchmark
  welford_benchmark.cpp
)

target_link_libraries(welford_benchmark
  PRIVATE
    ${CPP_LIBNAME}
)

add_executable(mse_benchmark
  mse_benchmark.cpp
)

target_link_libraries(mse_benchmark
  PRIVATE
    ${CPP_LIBNAME}
)
