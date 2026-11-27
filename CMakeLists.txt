cmake_minimum_required(VERSION 3.21)
# macOS builds need macOS 26 or newer and its SDK (src/gpu/gpu.mm: Metal 4); set before project() so it takes effect.
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum macOS version")
if(APPLE AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 26.0)
  message(WARNING "CMAKE_OSX_DEPLOYMENT_TARGET ${CMAKE_OSX_DEPLOYMENT_TARGET} is below macOS 26, which IndraNet needs "
                  "(src/gpu/gpu.mm uses Metal 4); building for 26.0 instead")
  set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum macOS version" FORCE)
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

# The library's layers, lowest first (docs/architecture.md). A layer may use only the layers below it; tfi_deps_<layer>
# lists them. The library is one archive of every layer; TFI_CHECK_LAYERS also builds each layer as a shared
# library linked only to the layers below it, so that any call into a layer above fails to link, and
# tools/check_layers.py (the ctest test "layers") checks the same of every #include.
set(TFI_LAYERS support text core gpu model analysis generation)
set(tfi_deps_support)
set(tfi_deps_text support)
set(tfi_deps_core support)
set(tfi_deps_gpu core support)
set(tfi_deps_model gpu core text support)
set(tfi_deps_analysis model gpu core text support)
set(tfi_deps_generation analysis model gpu core text support)
option(TFI_CHECK_LAYERS "Also link each layer of the library on its own, to check that none uses a layer above it" OFF)

set(TFI_SOURCES)
foreach(layer ${TFI_LAYERS})
  file(GLOB tfi_layer_sources CONFIGURE_DEPENDS src/${layer}/*.cpp)
  set(tfi_sources_${layer} ${tfi_layer_sources})
  list(APPEND TFI_SOURCES ${tfi_layer_sources})
endforeach()
file(GLOB tfi_stray CONFIGURE_DEPENDS src/*.cpp src/*/*.cpp)
list(REMOVE_ITEM tfi_stray ${TFI_SOURCES})
if(tfi_stray)
  message(FATAL_ERROR "Sources outside the library's layers (TFI_LAYERS): ${tfi_stray}")
endif()
add_library(tfi STATIC ${TFI_SOURCES})
add_library(tfi::tfi ALIAS tfi)  # the name find_package(tfi) gives, so either way of adding tfi links the same
target_compile_features(tfi PUBLIC cxx_std_20)
if(TFI_PYTHON OR SKBUILD)
  set_target_properties(tfi PROPERTIES POSITION_INDEPENDENT_CODE ON)  # linked into a shared module
endif()
target_include_directories(tfi PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(tfi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)  # internal headers: "model/ops.h", ...
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
    list(APPEND tfi_sources_gpu src/gpu/gpu.mm)
    set_source_files_properties(src/gpu/gpu.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_compile_definitions(tfi PUBLIC TFI_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
    target_link_libraries(tfi PUBLIC "-framework Accelerate" "-framework Metal" "-framework MetalPerformanceShaders"
                                     "-framework Foundation")
  endif()
endif()

if(TFI_CHECK_LAYERS)
  get_target_property(tfi_defs tfi INTERFACE_COMPILE_DEFINITIONS)
  get_target_property(tfi_libs tfi INTERFACE_LINK_LIBRARIES)
  foreach(layer ${TFI_LAYERS})
    add_library(tfi_layer_${layer} SHARED ${tfi_sources_${layer}})
    target_compile_features(tfi_layer_${layer} PRIVATE cxx_std_20)
    target_include_directories(tfi_layer_${layer} PRIVATE include src)
    target_compile_options(tfi_layer_${layer} PRIVATE -Wall -Wextra)
    if(tfi_defs)
      target_compile_definitions(tfi_layer_${layer} PRIVATE ${tfi_defs})
    endif()
    foreach(dep ${tfi_deps_${layer}})
      target_link_libraries(tfi_layer_${layer} PRIVATE tfi_layer_${dep})
    endforeach()
    target_link_libraries(tfi_layer_${layer} PRIVATE ${tfi_libs})
    if(NOT APPLE)  # macOS refuses undefined symbols in a shared library by default
      target_link_options(tfi_layer_${layer} PRIVATE -Wl,--no-undefined)
    endif()
  endforeach()
endif()

if(TFI_BUILD_TOOLS)
  add_executable(indra tools/indra.cpp)
  target_link_libraries(indra PRIVATE tfi)

  add_executable(tfi_golden tests/golden.cpp)
  target_link_libraries(tfi_golden PRIVATE tfi)

  file(GLOB tfi_unit_sources CONFIGURE_DEPENDS tests/unit/*.cpp)  # one file per group, and main.cpp
  add_executable(tfi_tests ${tfi_unit_sources})
  target_link_libraries(tfi_tests PRIVATE tfi)
  target_include_directories(tfi_tests PRIVATE src)  # the tests reach internals too

  enable_testing()
  # One test per group (tests/unit/<group>.cpp), so ctest -R gpu runs part and ctest -j runs them side by side.
  foreach(group core analysis grad fp16 gpu)
    add_test(NAME unit_${group} COMMAND tfi_tests --group ${group})
  endforeach()
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    add_test(NAME layers COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_layers.py
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
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
