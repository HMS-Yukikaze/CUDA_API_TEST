#ifndef _MCUDA_H_
#define _MCUDA_H_

#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <device_launch_parameters.h>

// Define image dimensions
const int imageWidth = 640;
const int imageHeight = 480;

// Define scale factor for resizing
const float scaleFactor = 0.5f;

// CUDA kernel for resizing image using affine transformation
__global__ void resizeImage(const uchar* input, uchar* output, int outputWidth, int outputHeight, int inputWidth, int inputHeight, float scaleX, float scaleY);


void resizeImageCUDA(const cv::Mat& inputImage, cv::Mat& outputImage, float scaleFactor);

#endif // !_MCUDA_H_
