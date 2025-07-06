set(CMAKE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pybind)

pybind11_add_module(smollnet ${CMAKE_SRC_DIR}/bindings.cpp)
target_link_libraries(smollnet PRIVATE ${CPP_LIBNAME})
