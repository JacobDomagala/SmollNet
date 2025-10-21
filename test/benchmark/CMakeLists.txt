add_executable(welford_benchmark
  welford_benchmark.cpp
)

target_link_libraries(welford_benchmark
  PRIVATE
    ${CPP_LIBNAME}
)
