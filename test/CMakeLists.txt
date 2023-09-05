find_package(GTest REQUIRED)

set(TEST_SOURCES 
    ../src/mcuda.cu
    test.cpp
)
add_executable(
    mtest
    ${TEST_SOURCES}
)

target_include_directories(mtest PRIVATE 
    ${OpenCVRoot}/include
    ${CMAKE_SOURCE_DIR}/include
)

# Link any additional libraries if needed
target_link_libraries(mtest
    ${OpenCVRoot}/x64/vc16/lib/opencv_cudawarping470.lib
    ${OpenCVRoot}/x64/vc16/lib/opencv_cudaimgproc470.lib
    ${OpenCVRoot}/x64/vc16/lib/opencv_core470.lib
    ${OpenCVRoot}/x64/vc16/lib/opencv_imgcodecs470.lib
    ${OpenCVRoot}/x64/vc16/lib/opencv_highgui470.lib
    ${OpenCVRoot}/x64/vc16/lib/opencv_imgproc470.lib
    GTest::GTest
    cudart.lib
)