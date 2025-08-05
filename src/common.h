#pragma once

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush" };

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0} };

#define CEIL(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }


struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const {
        if (obj) delete obj;
    } 
};

inline static auto StreamDeleter = [](cudaStream_t* pStream)
{
    if (pStream) {
        static_cast<void>(cudaStreamDestroy(*pStream));
        delete pStream;
    }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        pStream.reset(nullptr);
    }

    return pStream;
}

struct BufferDeleter 
{
    template <typename T>
    void operator()(T* ptr) const {
        if (ptr) {
            cudaFree(ptr);  
        }
    }
};

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

inline std::string getFilenameFromPath(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

inline std::string makeOutputPath(const std::string& inputPath, const std::string& newDir) {
    std::string filename = getFilenameFromPath(inputPath);
    if (newDir.back() == '/' || newDir.back() == '\\')
        return newDir + filename;
    else
        return newDir + "/" + filename;
}

// 将 float* (GPU) 中的数据保存为图片
inline void saveCudaImage(
    float* d_dst,          // CUDA memory: float[NCHW], RGB, [0,1]
    int width, int height, const std::string& save_path, cudaStream_t stream = nullptr) {

    size_t imageSize = width * height * 3;
    
    // 分配 host 内存（float 格式）
    std::vector<float> h_float(imageSize);

    // 从 device 拷贝到 host
    CUDA_CHECK(cudaMemcpyAsync(
        h_float.data(), d_dst,
        imageSize * sizeof(float),
        cudaMemcpyDeviceToHost, stream
    ));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // float[NCHW] -> uint8_t[HWC]
    cv::Mat image(height, width, CV_8UC3);
    int area = width * height;

    for (int y = 0; y < height; ++y) {
        uint8_t* ptr = image.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float r = h_float[idx + area * 0];
            float g = h_float[idx + area * 1];
            float b = h_float[idx + area * 2];

            // [0,1] -> [0,255] + clip
            ptr[x * 3 + 0] = static_cast<uint8_t>(std::min(std::max(b * 255.0f, 0.0f), 255.0f));
            ptr[x * 3 + 1] = static_cast<uint8_t>(std::min(std::max(g * 255.0f, 0.0f), 255.0f));
            ptr[x * 3 + 2] = static_cast<uint8_t>(std::min(std::max(r * 255.0f, 0.0f), 255.0f));
        }
    }

    cv::imwrite(save_path, image);
}
