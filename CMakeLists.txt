cmake_minimum_required(VERSION 3.21)
# macOS builds need macOS 15 or newer (src/gpu/gpu.mm); set before project() so it takes effect.
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version")
if(APPLE AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 15.0)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version" FORCE)
endif()
project(IndraNet VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TFI_PYTHON "Build the Python bindings (python/tfi; needs nanobind: pip install nanobind)" OFF)
# The CLI and the tests: by default only when this is the top-level project (not under add_subdirectory /
# FetchContent, nor pip's build of the Python module).
if(PROJECT_IS_TOP_LEVEL AND NOT SKBUILD)
  set(tfi_tools_default ON)
else()
  set(tfi_tools_default OFF)
endif()
option(TFI_BUILD_TOOLS "Build indra, tfi_golden and tfi_tests" ${tfi_tools_default})

file(GLOB TFI_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*/*.cpp)
add_library(tfi STATIC ${TFI_SOURCES})
add_library(tfi::tfi ALIAS tfi)  # the name find_package(tfi) gives, so either way of adding tfi links the same
target_compile_features(tfi PUBLIC cxx_std_20)
if(TFI_PYTHON OR SKBUILD)
  set_target_properties(tfi PROPERTIES POSITION_INDEPENDENT_CODE ON)  # linked into a shared module
endif()
target_include_directories(tfi PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_options(tfi PRIVATE -Wall -Wextra)
find_package(Threads REQUIRED)
target_link_libraries(tfi PUBLIC Threads::Threads)

# macOS: Accelerate for matrix multiplies, Metal for the GPU path of batched patching (src/gpu/gpu.mm).
# PUBLIC so the unit tests also exercise the macOS-only paths.
if(APPLE)
  find_library(ACCELERATE Accelerate)
  find_library(METAL Metal)
  find_library(MPS MetalPerformanceShaders)
  find_library(FOUNDATION Foundation)
  if(ACCELERATE AND METAL AND MPS AND FOUNDATION)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    target_sources(tfi PRIVATE src/gpu/gpu.mm)
    set_source_files_properties(src/gpu/gpu.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_compile_definitions(tfi PUBLIC TFI_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
    target_link_libraries(tfi PUBLIC "-framework Accelerate" "-framework Metal" "-framework MetalPerformanceShaders"
                                     "-framework Foundation")
  endif()
endif()

if(TFI_BUILD_TOOLS)
  add_executable(indra tools/indra.cpp)
  target_link_libraries(indra PRIVATE tfi)

  add_executable(tfi_golden tests/golden.cpp)
  target_link_libraries(tfi_golden PRIVATE tfi)

  add_executable(tfi_tests tests/unit.cpp)
  target_link_libraries(tfi_tests PRIVATE tfi)

  enable_testing()
  # One test per group (tests/unit.cpp), so ctest -R gpu runs part and ctest -j runs them side by side.
  foreach(group core analysis grad fp16 gpu)
    add_test(NAME unit_${group} COMMAND tfi_tests --group ${group})
  endforeach()
endif()

# cmake --install: the library, its headers and a CMake package, so that other projects can
# find_package(tfi) and link tfi::tfi (not for pip's build of the Python module, which installs only that).
if(NOT SKBUILD)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(TARGETS tfi EXPORT tfiTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY include/tfi DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT tfiTargets NAMESPACE tfi:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tfi)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tfiConfig.cmake
       "include(CMakeFindDependencyMacro)\nfind_dependency(Threads)\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/tfiTargets.cmake\")\n")
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/tfiConfigVersion.cmake COMPATIBILITY SameMinorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tfiConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/tfiConfigVersion.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tfi)
endif()

# Python bindings: the module _tfi. pip install (scikit-build-core, pyproject.toml) installs it with
# python/tfi/__init__.py as the package tfi; make python writes it next to python/tfi/__init__.py instead, so that
# `import tfi` finds it from the repository (pytest puts python/ on the path).
if(TFI_PYTHON OR SKBUILD)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT RESULT_VARIABLE nanobind_missing)
  if(nanobind_missing)
    message(FATAL_ERROR "TFI_PYTHON needs nanobind for ${Python_EXECUTABLE}: pip install nanobind")
  endif()
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_tfi NB_STATIC python/tfi/_tfi.cpp)
  target_link_libraries(_tfi PRIVATE tfi)
  if(SKBUILD)
    install(TARGETS _tfi LIBRARY DESTINATION tfi)
    install(FILES python/tfi/__init__.py DESTINATION tfi)
  else()
    set_target_properties(_tfi PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/tfi")
  endif()
endif()
