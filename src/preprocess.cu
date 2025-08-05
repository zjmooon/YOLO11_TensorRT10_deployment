#include "preprocess.h"
#include "common.h"

namespace preprocess {
TransInfo    trans;
AffineMatrix2 affine_matrix;

void warpaffine_init(int srcH, int srcW, int tarH, int tarW){
    trans.src_h = srcH;
    trans.src_w = srcW;
    trans.tar_h = tarH;
    trans.tar_w = tarW;
    affine_matrix.init(trans);
}

__host__ __device__ void affine_transformation(
    float trans_matrix[6], 
    int src_x, int src_y, 
    float* tar_x, float* tar_y)
{
    *tar_x = trans_matrix[0] * src_x + trans_matrix[1] * src_y + trans_matrix[2];
    *tar_y = trans_matrix[3] * src_x + trans_matrix[4] * src_y + trans_matrix[5];
}

__global__ void warpaffine_BGR2RGB_kernel(
    float* dst, uint8_t* src, 
    TransInfo trans,
    AffineMatrix2 affine_matrix)
{
    float src_x, src_y;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    affine_transformation(affine_matrix.reverse, x + 0.5, y + 0.5, &src_x, &src_y);

    int src_x1 = floor(src_x - 0.5);
    int src_y1 = floor(src_y - 0.5);
    int src_x2 = src_x1 + 1;
    int src_y2 = src_y1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y1 > trans.src_h || src_x1 > trans.src_w) {
    } else {
        float tw   = src_x - src_x1;
        float th   = src_y - src_y1;

        float a1_1 = (1.0 - tw) * (1.0 - th);
        float a1_2 = tw * (1.0 - th);
        float a2_1 = (1.0 - tw) * th;
        float a2_2 = tw * th;

        int srcIdx1_1 = (src_y1 * trans.src_w + src_x1) * 3;
        int srcIdx1_2 = (src_y1 * trans.src_w + src_x2) * 3;
        int srcIdx2_1 = (src_y2 * trans.src_w + src_x1) * 3;
        int srcIdx2_2 = (src_y2 * trans.src_w + src_x2) * 3;

        int tarIdx    = y * trans.tar_w  + x;
        int tarArea   = trans.tar_w * trans.tar_h;

        dst[tarIdx + tarArea * 0] = 
            round((a1_1 * src[srcIdx1_1 + 2] + 
                   a1_2 * src[srcIdx1_2 + 2] +
                   a2_1 * src[srcIdx2_1 + 2] +
                   a2_2 * src[srcIdx2_2 + 2])) / 255.0f;

        dst[tarIdx + tarArea * 1] = 
            round((a1_1 * src[srcIdx1_1 + 1] + 
                   a1_2 * src[srcIdx1_2 + 1] +
                   a2_1 * src[srcIdx2_1 + 1] +
                   a2_2 * src[srcIdx2_2 + 1])) / 255.0f;

        dst[tarIdx + tarArea * 2] = 
            round((a1_1 * src[srcIdx1_1 + 0] + 
                   a1_2 * src[srcIdx1_2 + 0] +
                   a2_1 * src[srcIdx2_1 + 0] +
                   a2_2 * src[srcIdx2_2 + 0])) / 255.0f;
    }
}

__global__ void bilinear_BGR2RGB_nhwc2nchw_shift_kernel(
    float* tar, uint8_t* src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
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

    if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW) {
        // bilinear interpolation -- 对于越界的坐标不进行计算
    } else {
        // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
        float th   = (float)y * scale - src_y1;
        float tw   = (float)x * scale - src_x1;

        // bilinear interpolation -- 计算面积
        float a1_1 = (1.0 - tw) * (1.0 - th);  // 右下
        float a1_2 = tw * (1.0 - th);          // 左下
        float a2_1 = (1.0 - tw) * th;          // 右上
        float a2_2 = tw * th;                  // 左上

        // bilinear interpolation -- 计算4个坐标所对应的索引
        int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;  // 左上
        int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;  // 右上
        int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;  // 左下
        int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;  // 右下

        // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
        y = y - int(srcH / (scale * 2)) + int(tarH / 2);
        x = x - int(srcW / (scale * 2)) + int(tarW / 2);

        // bilinear interpolation -- 计算resized之后的图的索引
        int tarIdx    = y * tarW  + x;
        int tarArea   = tarW * tarH;

        // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB + shift + nhwc2nchw
        tar[tarIdx + tarArea * 0] = 
            round((a1_1 * src[srcIdx1_1 + 2] + 
                   a1_2 * src[srcIdx1_2 + 2] +
                   a2_1 * src[srcIdx2_1 + 2] +
                   a2_2 * src[srcIdx2_2 + 2])) / 255.0f;

        tar[tarIdx + tarArea * 1] = 
            round((a1_1 * src[srcIdx1_1 + 1] + 
                   a1_2 * src[srcIdx1_2 + 1] +
                   a2_1 * src[srcIdx2_1 + 1] +
                   a2_2 * src[srcIdx2_2 + 1])) / 255.0f;

        tar[tarIdx + tarArea * 2] = 
            round((a1_1 * src[srcIdx1_1 + 0] + 
                   a1_2 * src[srcIdx1_2 + 0] +
                   a2_1 * src[srcIdx2_1 + 0] +
                   a2_2 * src[srcIdx2_2 + 0])) / 255.0f;
    }
}

struct AffineMatrix {
    float value[6];
};

__global__ void warpaffine_kernel(
    uint8_t* src, int src_line_size, int src_width,
    int src_height, float* dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    else {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = { const_value_st, const_value_st, const_value_st };
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // bgr to rgb 
    float t = c2;
    c2 = c0;
    c0 = t;

    // normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    // rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void cuda_preprocess(cv::Mat &h_src, float* d_dst, int dst_h, int dst_w, cudaStream_t stream) {
    uint8_t* d_src = nullptr;

    // input image hw
    int height   = h_src.rows;
    int width    = h_src.cols;
    int chan     = 3;

    int src_size  = height * width * chan * sizeof(uint8_t);

    // 分配device上的src和tar的内存
    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    
    // CUDA_CHECK(cudaMemcpyAsync(d_src, h_src.ptr(), src_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_src, h_src.data, src_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // device上处理resize, BGR2RGB的核函数
    // resize_bilinear_gpu(d_dst, d_src, dst_h, dst_w, width, height, stream);
    dim3 dimBlock(32, 32);
    dim3 dimGrid(CEIL(dst_w, 32), CEIL(dst_h, 32));
   
    if (true) {
        warpaffine_init(height, width, dst_h, dst_w);
        warpaffine_BGR2RGB_kernel <<<dimGrid, dimBlock, 0, stream>>> 
        (d_dst, d_src, trans, affine_matrix);  
        // save preprocess
        if (false)  saveCudaImage(d_dst, dst_w, dst_h, "../results/preprocess.jpg", stream); 
         
    }
    else {
        //scaled resize
        float scaled_h = (float)height / dst_h;
        float scaled_w = (float)width / dst_w;
        float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

        bilinear_BGR2RGB_nhwc2nchw_shift_kernel 
                    <<<dimGrid, dimBlock, 0, stream>>> 
                    (d_dst, d_src, dst_w, dst_h, width, height, scale);    
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_src));

    // cv::Mat h_tar(cv::Size(dst_h, dst_w), CV_8UC3); 
    // CUDA_CHECK(cudaMemcpy(h_tar.data, , dst_h * dst_w * 3, cudaMemcpyDeviceToHost));
    // cv::cvtColor(h_tar, h_tar, cv::COLOR_RGB2BGR);
    // cv::imwrite("output_path.png", h_tar);
    // 数据已经拷贝到了 gpu_input_中，可以进行enqueueV3
}

static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;

void cuda_preprocess(uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height, cudaStream_t stream)
{
    int img_size = src_width * src_height * 3;
    // copy data to pinned memory
    memcpy(img_buffer_host, src, img_size);
    // copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, stream));

    AffineMatrix s2d, d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);

    warpaffine_kernel << <blocks, threads, 0, stream >> > (
        img_buffer_device, src_width * 3, src_width,
        src_height, dst, dst_width,
        dst_height, 128, d2s, jobs);

}

void cuda_preprocess_init(int max_image_size) {
    // prepare input data in pinned memory
    CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));
    // prepare input data in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy() {
    CUDA_CHECK(cudaFree(img_buffer_device));
    CUDA_CHECK(cudaFreeHost(img_buffer_host));
}

} // namespace preprocess