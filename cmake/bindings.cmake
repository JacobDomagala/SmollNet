set(CMAKE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)

pybind11_add_module(smollnet ${CMAKE_SRC_DIR}/bindings.cpp)
target_link_libraries(smollnet PRIVATE ${CPP_LIBNAME})

set_target_properties(smollnet PROPERTIES
  PREFIX ""
)
install(TARGETS smollnet
  LIBRARY DESTINATION smollnet
  RUNTIME DESTINATION smollnet
)
install(DIRECTORY python/smollnet
  DESTINATION .
)
