#include "mcuda.cuh"

// CUDA kernel for resizing image using affine transformation
__global__ void resizeImage(const uchar* input, uchar* output, int outputWidth, int outputHeight, int inputWidth, int inputHeight, float scaleX, float scaleY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputWidth && y < outputHeight)
    {
        float srcX = (x + 0.5f) / scaleX - 0.5f;
        float srcY = (y + 0.5f) / scaleY - 0.5f;

        int srcX0 = static_cast<int>(srcX);
        int srcX1 = srcX0 + 1;
        int srcY0 = static_cast<int>(srcY);
        int srcY1 = srcY0 + 1;

        float xWeight = srcX - srcX0;
        float yWeight = srcY - srcY0;

        uchar* outputPixel = output + (y * outputWidth + x) * 3;
        const uchar* inputPixel00 = input + (srcY0 * inputWidth + srcX0) * 3;
        const uchar* inputPixel01 = input + (srcY0 * inputWidth + srcX1) * 3;
        const uchar* inputPixel10 = input + (srcY1 * inputWidth + srcX0) * 3;
        const uchar* inputPixel11 = input + (srcY1 * inputWidth + srcX1) * 3;

        for (int channel = 0; channel < 3; ++channel)
        {
            float top = static_cast<float>(inputPixel00[channel]) * (1 - xWeight) + static_cast<float>(inputPixel01[channel]) * xWeight;
            float bottom = static_cast<float>(inputPixel10[channel]) * (1 - xWeight) + static_cast<float>(inputPixel11[channel]) * xWeight;
            outputPixel[channel] = static_cast<uchar>(top * (1 - yWeight) + bottom * yWeight);
        }
    }
}


void resizeImageCUDA(const cv::Mat& inputImage, cv::Mat& outputImage, float scaleFactor)
{
    // Allocate memory for input and output images on the GPU
    uchar* d_inputImage, * d_outputImage;
    cudaMalloc((void**)&d_inputImage, inputImage.cols * inputImage.rows * inputImage.channels());
    cudaMalloc((void**)&d_outputImage, outputImage.cols * outputImage.rows * inputImage.channels());

    // Copy input image to GPU memory
    cudaMemcpy(d_inputImage, inputImage.data, inputImage.cols * inputImage.rows * inputImage.channels(), cudaMemcpyHostToDevice);

    // Define CUDA block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((outputImage.cols + blockDim.x - 1) / blockDim.x, (outputImage.rows + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel to resize the image
    resizeImage << <gridDim, blockDim >> > (d_inputImage, d_outputImage, outputImage.cols, outputImage.rows, inputImage.cols, inputImage.rows, scaleFactor, scaleFactor);

    // Copy the resized image back to the CPU
    cudaMemcpy(outputImage.data, d_outputImage, outputImage.cols * outputImage.rows * inputImage.channels(), cudaMemcpyDeviceToHost);

    // Cleanup GPU memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}
