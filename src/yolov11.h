#ifndef YOLO11_H
#define YOLO11_H

#include <opencv2/opencv.hpp>
#include "common.h"

struct Detection
{
    float x0, y0, x1, y1, conf;
    int class_id;
};

class YOLOv11
{

public:

    YOLOv11(std::string model_path);
    ~YOLOv11();

    void infer(cv::Mat &img);
    std::string input_image_path_;
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
    void postprocess();
    void draw(cv::Mat& image);
    void parseDetections(const float* output, std::vector<Detection>& result);

    // Model parameters
    int input_w;
    int input_h;
    int num_detections;
    int detection_attribute_size;
    float conf_threshold_ = 0.4f;
    std::vector<Detection> detections_boxes_;
    

    std::vector<cv::Scalar> colors;

};

#endif