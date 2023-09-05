cmake_minimum_required(VERSION 3.17)

SET(CMAKE_TOOLCHAIN_FILE "D:/1_workSpace/vcpkg/scripts/buildsystems/vcpkg.cmake")

project(cuda_test LANGUAGES CUDA CXX)

# Specify the CUDA version (optional but recommended)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75") # Replace 'XX' with the compute capability of your GPU, e.g., 52, 60, 75, etc.

set(OpenCVRoot D:/1_workSpace/3rd/opencv/build/install)

# Set CUDA architectures (change as needed)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

file(GLOB_RECURSE ALL_SOURCES
    "${CMAKE_SOURCE_DIR}/include/*.h"
    "${CMAKE_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_SOURCE_DIR}/include/*.cuh"
    "${CMAKE_SOURCE_DIR}/src/*.c"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
    "${CMAKE_SOURCE_DIR}/src/*.cpp"    
)

# Filter out files from the build directory
#list(FILTER ALL_SOURCES EXCLUDE REGEX ".*build/.*")

source_group(TREE "${CMAKE_SOURCE_DIR}" FILES ${ALL_SOURCES})

# Add your executable
add_executable(cuda_test ${ALL_SOURCES})

target_include_directories(cuda_test PRIVATE 
    ${OpenCVRoot}/include
    ${CMAKE_SOURCE_DIR}/include
)

# Link any additional libraries if needed
target_link_libraries(cuda_test
    ${OpenCVRoot}/x64/vc16/lib/opencv_cudawarping470.lib
    ${OpenCVRoot}/x64/vc16/lib/opencv_cudaimgproc470.lib
    ${OpenCVRoot}/x64/vc16/lib/opencv_core470.lib
    ${OpenCVRoot}/x64/vc16/lib/opencv_imgcodecs470.lib
    ${OpenCVRoot}/x64/vc16/lib/opencv_highgui470.lib
    ${OpenCVRoot}/x64/vc16/lib/opencv_imgproc470.lib
    cudart.lib
)

# Set compiler flags if necessary
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Specify the output directory for the binary
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Specify the output directory for CUDA files (optional)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cuda)

# Optionally, enable/disable some features
# For example, to enable/disable debug mode:
# set(CMAKE_BUILD_TYPE Debug)
enable_testing()
add_subdirectory(test)