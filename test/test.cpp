#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <mcuda.cuh>
#include <iostream>
#include <chrono>


TEST(ResizeImageCUDATest_LIMIT_200ms, ExecutionTime)
{
    // Load the input image
    cv::Mat inputImage = cv::imread("input.jpg");

    // Define scale factor for resizing
    const float scaleFactor = 0.5f;

    // Calculate dimensions of the output image
    int outputWidth = static_cast<int>(inputImage.cols * scaleFactor);
    int outputHeight = static_cast<int>(inputImage.rows * scaleFactor);

    // Create an output image
    cv::Mat outputImage(outputHeight, outputWidth, inputImage.type());

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    // Resize the image using the CUDA function
    resizeImageCUDA(inputImage, outputImage, scaleFactor);
    auto end = std::chrono::high_resolution_clock::now();

    auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Perform your assertions, e.g., check if execution time is below a threshold
    EXPECT_LT(cost, 200); // Adjust the threshold as needed
}

TEST(ResizeImageCUDATest_LIMIT_100ms, ExecutionTime)
{
    // Load the input image
    cv::Mat inputImage = cv::imread("input1.jpg");

    // Define scale factor for resizing
    const float scaleFactor = 0.5f;

    // Calculate dimensions of the output image
    int outputWidth = static_cast<int>(inputImage.cols * scaleFactor);
    int outputHeight = static_cast<int>(inputImage.rows * scaleFactor);

    // Create an output image
    cv::Mat outputImage(outputHeight, outputWidth, inputImage.type());

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    // Resize the image using the CUDA function
    resizeImageCUDA(inputImage, outputImage, scaleFactor);
    auto end = std::chrono::high_resolution_clock::now();

    auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Perform your assertions, e.g., check if execution time is below a threshold
    EXPECT_LT(cost, 100); // Adjust the threshold as needed
}

TEST(ResizeImageCUDATest_LIMIT_50ms, ExecutionTime)
{
    // Load the input image
    cv::Mat inputImage = cv::imread("input2.jpg");

    // Define scale factor for resizing
    const float scaleFactor = 0.5f;

    // Calculate dimensions of the output image
    int outputWidth = static_cast<int>(inputImage.cols * scaleFactor);
    int outputHeight = static_cast<int>(inputImage.rows * scaleFactor);

    // Create an output image
    cv::Mat outputImage(outputHeight, outputWidth, inputImage.type());

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    // Resize the image using the CUDA function
    resizeImageCUDA(inputImage, outputImage, scaleFactor);
    auto end = std::chrono::high_resolution_clock::now();

    auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Perform your assertions, e.g., check if execution time is below a threshold
    EXPECT_LT(cost, 50); // Adjust the threshold as needed
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}