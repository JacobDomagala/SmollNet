include(GNUInstallDirs)

set(SMOLLNET_CUDA_ROOT "${CUDAToolkit_ROOT}")
set(SMOLLNET_FMT_ROOT "${fmt_DIR}")

install(TARGETS ${PROJECT_NAME}
        EXPORT SmollNetTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY src/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cuh")

install(EXPORT SmollNetTargets
        FILE SmollNetTargets.cmake
        NAMESPACE SmollNet::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmollNet)

include(CMakePackageConfigHelpers)
configure_package_config_file(
        cmake/SmollNetConfig.cmake.in SmollNetConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmollNet)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SmollNetConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SmollNet)
