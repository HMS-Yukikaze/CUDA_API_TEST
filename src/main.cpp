#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "mcuda.cuh"


using namespace std;
using namespace cv;


//bool check_runtime(cudaError_t e, const char* call, int line, const char* file) {
//    if (e != cudaSuccess) {
//        printf("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
//        return false;
//    }
//    return true;
//}
//
//void warp_affine_bilinear(uint8_t* src,int src_width,int src_height,
//                          uint8_t* dst, int dst_width, int dst_height);


//Mat warpAffine2align(Mat img,const Size &sz) {
//    Mat output(sz,CV_8UC3);
//    
//    uint8_t* psrc_dev = nullptr;
//    uint8_t* pdst_dev = nullptr;
//
//    size_t src_size = img.cols * img.rows;
//    size_t dst_size = sz.width * sz.height;
//
//    check_runtime(cudaMalloc(&psrc_dev,src_size),__func__,__LINE__,__FILE__);
//    check_runtime(cudaMalloc(&pdst_dev,dst_size), __func__, __LINE__, __FILE__);
//
//    check_runtime(cudaMemcpy(&psrc_dev,img.data,src_size,cudaMemcpyHostToDevice), __func__, __LINE__, __FILE__);
//
//    warp_affine_bilinear(psrc_dev,img.cols,img.rows,
//                         pdst_dev,sz.width,sz.height);
//
//    check_runtime(cudaMemcpy(output.data,pdst_dev,dst_size, cudaMemcpyHostToDevice), __func__, __LINE__, __FILE__);
//
//    check_runtime(cudaFree(pdst_dev), __func__, __LINE__, __FILE__);
//    check_runtime(cudaFree(psrc_dev), __func__, __LINE__, __FILE__);
//    
//    return output;
//}





int main() {
    // Load input image (assumes you have a valid image file)
    cv::Mat inputImage = cv::imread("input.jpg");
    if (inputImage.empty())
    {
        std::cerr << "Error: Unable to load input image." << std::endl;
        return -1;
    }

    // Define scale factor for resizing
    const float scaleFactor = 0.5f;

    // Calculate dimensions of the output image
    int outputWidth = static_cast<int>(inputImage.cols * scaleFactor);
    int outputHeight = static_cast<int>(inputImage.rows * scaleFactor);

    // Create an output image
    cv::Mat outputImage(outputHeight, outputWidth, inputImage.type());

    auto start=std::chrono::high_resolution_clock::now();

    // Resize the image using the CUDA function
    resizeImageCUDA(inputImage, outputImage, scaleFactor);

    auto end = std::chrono::high_resolution_clock::now();

    auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << cost << std::endl;

    // Save the resized image
    cv::imwrite("output.jpg", outputImage);

  


    return 0;
}
