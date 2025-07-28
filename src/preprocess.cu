#include "preprocess.h"
#include "common.h"

__global__ void bilinear_BGR2RGB_nhwc2nchw_shift_norm_kernel(
    float* tar, uint8_t* src, 
    int dst_h, int dst_w, 
    int src_h, int src_w, 
    float scale) 
{
    // resized之后的图tar上的坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
    int src_y1 = floor((y + 0.5) * scale - 0.5);
    int src_x1 = floor((x + 0.5) * scale - 0.5);
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y2 > src_h || src_x2 > src_w) {
        // bilinear interpolation -- 对于越界的坐标不进行计算
    } else {
        // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
        float th   = (float)y * scale - src_y1;
        float tw   = (float)x * scale - src_x1;

        // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
        float a1_1 = (1.0 - tw) * (1.0 - th);  // 右下
        float a1_2 = tw * (1.0 - th);          // 左下
        float a2_1 = (1.0 - tw) * th;          // 右上
        float a2_2 = tw * th;                  // 左上

        // bilinear interpolation -- 计算4个坐标所对应的索引
        int srcIdx1_1 = (src_y1 * src_w + src_x1) * 3;  // 左上
        int srcIdx1_2 = (src_y1 * src_w + src_x2) * 3;  // 右上
        int srcIdx2_1 = (src_y2 * src_w + src_x1) * 3;  // 左下
        int srcIdx2_2 = (src_y2 * src_w + src_x2) * 3;  // 右下

        // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
        y = y - int(src_h / (scale * 2)) + int(dst_h / 2);
        x = x - int(src_w / (scale * 2)) + int(dst_w / 2);

        // bilinear interpolation -- 计算resized之后的图的索引
        int tarIdx    = (y * dst_w  + x) * 3;
        int tarArea   = dst_w * dst_h;

        // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB + shift + nhwc2nchw(rgbrgbrgb->rrrrgggbb)
        tar[tarIdx + tarArea * 0] = 
            (round((a1_1 * src[srcIdx1_1 + 2] + 
                   a1_2 * src[srcIdx1_2 + 2] +
                   a2_1 * src[srcIdx2_1 + 2] +
                   a2_2 * src[srcIdx2_2 + 2])) / 255.0f);

        tar[tarIdx + tarArea * 1] = 
            (round((a1_1 * src[srcIdx1_1 + 1] + 
                   a1_2 * src[srcIdx1_2 + 1] +
                   a2_1 * src[srcIdx2_1 + 1] +
                   a2_2 * src[srcIdx2_2 + 1])) / 255.0f);

        tar[tarIdx + tarArea * 2] = 
            (round((a1_1 * src[srcIdx1_1 + 0] + 
                   a1_2 * src[srcIdx1_2 + 0] +
                   a2_1 * src[srcIdx2_1 + 0] +
                   a2_2 * src[srcIdx2_2 + 0])) / 255.0f);
    }
}

void preprocess_resize_gpu(cv::Mat &h_src, float *d_dst, int dst_h, int dst_w, cudaStream_t stream) {
    uint8_t* d_src  = nullptr;

    // input image hw
    int height   = h_src.rows;
    int width    = h_src.cols;
    int chan     = 3;

    int src_size  = height * width * chan * sizeof(uint8_t);
    int norm_size = 3 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    CUDA_CHECK(cudaMemcpyAsync(d_src, h_src.data, src_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // device上处理resize, BGR2RGB的核函数
    resize_bilinear_gpu(d_dst, d_src, dst_h, dst_w, width, height, stream);

    CUDA_CHECK(cudaFree(d_src));

    // 数据已经拷贝到了 gpu_input_中，可以进行enqueueV3
}

void resize_bilinear_gpu(float *d_dst, uint8_t *d_src, int dst_h, int dst_w, int src_h, int src_w, cudaStream_t stream)
{

    dim3 dimBlock(32, 32, 1);
    // dim3 dimGrid(dst_w / 32 + 1, dst_h / 32 + 1, 1);
    dim3 dimGrid(CEIL(dst_w, 32), CEIL(dst_h, 32));
   
    //scaled resize
    float scaled_h = (float)src_h / dst_h;
    float scaled_w = (float)src_w / dst_w;
    float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

    bilinear_BGR2RGB_nhwc2nchw_shift_norm_kernel 
                <<<dimGrid, dimBlock, 0, stream>>> 
                (d_dst, d_src, dst_w, dst_h, src_w, src_h, scale);
}
