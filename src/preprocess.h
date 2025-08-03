#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

struct TransInfo{
    int src_w = 0;
    int src_h = 0;
    int tar_w = 0;
    int tar_h = 0;
    TransInfo() = default;
    TransInfo(int srcW, int srcH, int tarW, int tarH):
        src_w(srcW), src_h(srcH), tar_w(tarW), tar_h(tarH){}
};

struct AffineMatrix2{
    float forward[6];
    float reverse[6];
    float forward_scale;
    float reverse_scale;

    void calc_forward_matrix(TransInfo trans){
        forward[0] = forward_scale;
        forward[1] = 0;
        forward[2] = - forward_scale * trans.src_w * 0.5 + trans.tar_w * 0.5;
        forward[3] = 0;
        forward[4] = forward_scale;
        forward[5] = - forward_scale * trans.src_h * 0.5 + trans.tar_h * 0.5;
    };

    void calc_reverse_matrix(TransInfo trans){
        reverse[0] = reverse_scale;
        reverse[1] = 0;
        reverse[2] = - reverse_scale * trans.tar_w * 0.5 + trans.src_w * 0.5;
        reverse[3] = 0;
        reverse[4] = reverse_scale;
        reverse[5] = - reverse_scale * trans.tar_h * 0.5 + trans.src_h * 0.5;
    };

    void init(TransInfo trans){
        float scaled_w = (float)trans.tar_w / trans.src_w;
        float scaled_h = (float)trans.tar_h / trans.src_h;
        forward_scale = (scaled_w < scaled_h ? scaled_w : scaled_h);
        reverse_scale = 1 / forward_scale;
    
        calc_forward_matrix(trans);
        calc_reverse_matrix(trans);
    }
};

namespace preprocess{

extern  TransInfo    trans;
extern  AffineMatrix2 affine_matrix;

void preprocess_resize_gpu(cv::Mat &h_src, float* d_dst, int dst_h, int dst_w, cudaStream_t stream);

void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();
void cuda_preprocess(uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream);

__host__ __device__ void affine_transformation(float* trans_matrix, int src_x, int src_y, float* tar_x, float* tar_y);
} // namespace preprocess