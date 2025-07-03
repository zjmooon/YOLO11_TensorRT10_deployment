
#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const {
        if (obj) delete obj;
    } 
};

static auto StreamDeleter = [](cudaStream_t* pStream)
{
    if (pStream) {
        static_cast<void>(cudaStreamDestroy(*pStream));
        delete pStream;
    }
}

struct BufferDeleter
{
    template <typename T>
    void operator()(T* obj) const {
        if (obj != nullptr) {
            cudaFree(obj);
            a = nullptr;
        }
    } 
};
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const {
        if (obj) delete obj;
    } 
};

struct Detection
{
    float conf;
    int class_id;
    Rect bbox;
};

class YOLOv11
{

public:

    YOLOv11(string model_path, nvinfer1::ILogger& logger);
    ~YOLOv11();

    void preprocess(Mat& image);
    void infer();
    void postprocess(vector<Detection>& output);
    void draw(Mat& image, const vector<Detection>& output);

private:
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    // float* gpu_buffers[2];               //!< The vector of device buffers needed for engine execution
    // float* cpu_output_buffer;

    // GPU
    std::unique_ptr<float, BufferDeleter> gpu_input_;
    std::unique_ptr<float, BufferDeleter> gpu_output_;
    std::vector<float>cpu_output;

    // TensorRT
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context_;
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> stream_(new cudaStream_t, StreamDeleter);

    cudaStream_t stream = nullptr;
    // IRuntime* runtime;                 //!< The TensorRT runtime used to deserialize the engine
    // ICudaEngine* engine;               //!< The TensorRT engine used to run the network
    // IExecutionContext* context;        //!< The context for executing inference using an ICudaEngine

    // Model parameters
    int input_w;
    int input_h;
    int num_detections;
    int detection_attribute_size;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    float conf_threshold = 0.4f;

    vector<Scalar> colors;

};
