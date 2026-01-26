#include "yolov11.h"
#include "preprocess.h"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>


static Logger logger;
constexpr int MAX_IMAGE_SIZE = 4096 * 4096;
constexpr int VERSION = 2;

YOLOv11::YOLOv11(std::string model_path)
{
    init(model_path);
}

void YOLOv11::init(std::string engine_path)
{
    // Read the engine file
    std::ifstream engineStream(engine_path, std::ios::binary);
    if (!engineStream.good()) {
        std::cout << "Error reading engine file: " << engine_path << std::endl;
        return;
    }
    engineStream.seekg(0, std::ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // For runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (runtime_.get() == nullptr) {
        std::cout << "Failed to create infer runtime" << std::endl;
        return;
    }
    // For engine
    engine_.reset(runtime_->deserializeCudaEngine(engineData.get(), modelSize));
    if (engine_.get() == nullptr) {
        std::cout << "Failed to create infer engine" << std::endl;
        return;
    }
    // For context 
    context_.reset(engine_->createExecutionContext());
    if (context_.get() == nullptr) {
        std::cout << "Failed to create infer context" << std::endl;
        return;
    }

    // Get input and output sizes of the model
    const char* inputName = engine_->getIOTensorName(0);
    int dimNums = engine_->getNbIOTensors();
    const char* outputName = engine_->getIOTensorName(dimNums - 1);
    nvinfer1::Dims input_dims = engine_->getTensorShape(inputName);
    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];
    nvinfer1::Dims output_dims = engine_->getTensorShape(outputName); //
    num_detections = output_dims.d[1];
    detection_attribute_size = output_dims.d[2];

    // print model input & output
    /* std::cout << "IOTensors num: " << engine_->getNbIOTensors() << std::endl;   // 打印模型所有dims包括输入输出
    inputName=images, input_dims.nbDims=4
    std::cout << "inputName=" << inputName << ", input_dims.nbDims=" << input_dims.nbDims << std::endl;    // 输入张量的元素的个数 BCHW
    for (const auto &x : input_dims.d ) {
        std::cout << x << " "; // 1 3 640 640
    }
    std::cout << std::endl;
    outputName=output0, output_dims.nbDims=3
    std::cout << "outputName=" << outputName << ", output_dims.nbDims=" << output_dims.nbDims << std::endl;
    for (const auto &x : output_dims.d) {
        std::cout << x << " ";  // 1 84 8400 or 1x300x6
    }
    std::cout << std::endl; */

    // For cpu output buffer, gpu input&output buffer  
    cpu_outpu_.resize(num_detections * detection_attribute_size);

    gpu_input_.reset(nullptr);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu_input_), 3 * input_w_ * input_h_ * sizeof(float)));

    gpu_output_.reset(nullptr);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu_output_), detection_attribute_size * num_detections * sizeof(float)));
    
    detections_boxes_.reserve(num_detections);

    if (VERSION == 2) preprocess::cuda_preprocess_init(MAX_IMAGE_SIZE);
}

YOLOv11::~YOLOv11(){
    if (VERSION == 2) preprocess::cuda_preprocess_destroy();
}

void YOLOv11::preprocess(cv::Mat& image) {
    if (VERSION == 1) {
        preprocess::cuda_preprocess(image, gpu_input_.get(), input_h_, input_w_, *stream_);
    }
    else if (VERSION == 2) {
        preprocess::cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_input_.get(), input_w_, input_h_, *stream_);
    }
}

void YOLOv11::infer(cv::Mat &img)
{
    preprocess(img);

    // binding GPU input&output
    context_->setTensorAddress(engine_->getIOTensorName(0), gpu_input_.get());
    context_->setTensorAddress(engine_->getIOTensorName(engine_->getNbIOTensors() - 1), gpu_output_.get());
    context_->enqueueV3(*stream_);
    CUDA_CHECK(cudaStreamSynchronize(*stream_));

    postprocess();
    drawBoxesOnCpu(img);
}

void YOLOv11::postprocess()
{
    // Memcpy from device output buffer to host output buffer
    if (true) {
        cudaMemcpyAsync(cpu_outpu_.data(), gpu_output_.get(), num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, *stream_);
        cudaStreamSynchronize(*stream_);
    } 
    else { // 隐式同步
        cudaMemcpy(cpu_outpu_.data(), gpu_output_.get(), num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    parseDetections(cpu_outpu_.data(), detections_boxes_);

    // for (const auto& x : detections_boxes_) {
    //     std::cout << x.x0   << " " << x.y0 << " " << x.x1 << " " << x.y1 << " " 
    //               << x.conf << " " << x.class_id  << std::endl;
    // }
}

void YOLOv11::drawBoxesOnCpu(cv::Mat& image)
{
    for (int i = 0; i < detections_boxes_.size(); i++)
    {
        Detection detection = detections_boxes_[i];
        int class_id = detection.class_id;
        float conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);
        
        rectangle(image, cv::Point(detection.x0, detection.y0), cv::Point(detection.x1, detection.y1), color, 3);

        // Detection box text
        std::string class_string = CLASS_NAMES[class_id] + ' ' + std::to_string(conf).substr(0, 4);
        cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_rect(detection.x0, detection.y0 - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, cv::FILLED);
        putText(image, class_string, cv::Point(detection.x0 + 5, detection.y0 - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }

    // save infered img
    // std::string result_path = makeOutputPath(input_image_path_, "../results");
    // // result_path = "../../results/car.jpg"
    // cv::imwrite("ggg.jpg" , image);
    detections_boxes_.clear();
}

void YOLOv11::parseDetections(const float *output, std::vector<Detection> &detections)
{
    // const float* data = output + num_detections * detection_attribute_size;
    for (int i = 0; i < num_detections; ++i) {
        // const float* row = data + i * detection_attribute_size;
        const float* detection = output + i * detection_attribute_size;
        float conf = detection[4];
        if (conf < conf_threshold_) continue;
        
        float x0 = detection[0];
        float y0 = detection[1];
        float x1 = detection[2];
        float y1 = detection[3];

        if (false) {
            const float cx = detection[0];
            const float cy = detection[1];
            const float ow = detection[2];
            const float oh = detection[3];
            x0 = static_cast<int>((cx));
            y0 = static_cast<int>((cy));
            x1 = static_cast<int>(x0 + ow);
            y1 = static_cast<int>(y0 + oh);
        }

        int cls = static_cast<int>(detection[5]);

        preprocess::affine_transformation(preprocess::affine_matrix.reverse, x0, y0, &x0, &y0);
        preprocess::affine_transformation(preprocess::affine_matrix.reverse, x1, y1, &x1, &y1);
        
        detections.push_back({x0, y0, x1, y1, conf, cls});
    }

}
