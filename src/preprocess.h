#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

void preprocess_resize_gpu(cv::Mat &h_src, float* d_dst, int dst_h, int dst_w, cudaStream_t stream);
void resize_bilinear_gpu(float* d_dst, uint8_t* d_src, int dst_h, int dst_w, int width, 
    int height, cudaStream_t stream);