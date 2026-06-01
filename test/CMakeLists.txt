add_subdirectory(benchmark)

add_executable(dataset_loader_test
  dataset_loader_test.cpp
)

target_link_libraries(dataset_loader_test
  PRIVATE
    ${CPP_LIBNAME}
)
