cmake_minimum_required(VERSION 3.16)
# macOS builds need macOS 15 or newer (src/gpu/gpu.mm); set before project() so it takes effect.
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version")
project(IndraNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TFI_PYTHON "Build the Python bindings (python/tfi; needs nanobind: pip install nanobind)" OFF)

file(GLOB TFI_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*/*.cpp)
add_library(tfi STATIC ${TFI_SOURCES})
if(TFI_PYTHON)
  set_target_properties(tfi PROPERTIES POSITION_INDEPENDENT_CODE ON)  # linked into a shared module
endif()
target_include_directories(tfi PUBLIC include)
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
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    target_sources(tfi PRIVATE src/gpu/gpu.mm)
    set_source_files_properties(src/gpu/gpu.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_compile_definitions(tfi PUBLIC TFI_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
    target_link_libraries(tfi PUBLIC ${ACCELERATE} ${METAL} ${MPS} ${FOUNDATION})
  endif()
endif()

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

# Python bindings: the module _tfi, written next to python/tfi/__init__.py so that `import tfi` finds it
# (pytest puts python/ on the path). make python runs this.
if(TFI_PYTHON)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT RESULT_VARIABLE nanobind_missing)
  if(nanobind_missing)
    message(FATAL_ERROR "TFI_PYTHON needs nanobind for ${Python_EXECUTABLE}: pip install nanobind")
  endif()
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_tfi NB_STATIC python/tfi/_tfi.cpp)
  target_link_libraries(_tfi PRIVATE tfi)
  set_target_properties(_tfi PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/tfi")
endif()
