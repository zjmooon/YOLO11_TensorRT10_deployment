#ifndef YOLO11_H
#define YOLO11_H

#include <opencv2/opencv.hpp>
#include "common.h"

struct Detection
{
    float conf;
    int class_id;
    cv::Rect bbox;
};

class YOLOv11
{

public:

    YOLOv11(std::string model_path);
    ~YOLOv11();

    void infer(cv::Mat& image);
    void postprocess(std::vector<Detection>& output);
    void draw(cv::Mat& image, const std::vector<Detection>& output);

private:
    void init(std::string engine_path);

    // GPU
    std::unique_ptr<float, BufferDeleter> gpu_input_;
    std::unique_ptr<float, BufferDeleter> gpu_output_;
    std::vector<float> cpu_outpu_;

    // TensorRT
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context_;
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> stream_ = makeCudaStream();

    void preprocess(cv::Mat& image);

    // Model parameters
    int input_w;
    int input_h;
    int num_detections;
    int detection_attribute_size;
    float conf_threshold = 0.4f;

    std::vector<cv::Scalar> colors;

};

#endif