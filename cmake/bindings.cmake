set(CMAKE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pybind)

pybind11_add_module(tensor ${CMAKE_SRC_DIR}/bindings.cpp)
target_link_libraries(tensor PRIVATE ${CPP_LIBNAME})
