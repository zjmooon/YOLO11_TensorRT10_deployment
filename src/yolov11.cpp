#include "yolov11.h"
#include "preprocess.h"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>


static Logger logger;

YOLOv11::YOLOv11(std::string model_path)
{
    init(model_path);
}


void YOLOv11::init(std::string engine_path)
{
    // Read the engine file
    std::ifstream engineStream(engine_path, std::ios::binary);
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
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    nvinfer1::Dims output_dims = engine_->getTensorShape(outputName); //
    num_detections = output_dims.d[1];
    detection_attribute_size = output_dims.d[2];

    std::cout << engine_->getNbIOTensors() << std::endl;   // 打印模型所有dims包括输入输出

    // inputName=images, input_dims.nbDims=4
    std::cout << "inputName=" << inputName << ", input_dims.nbDims=" << input_dims.nbDims << std::endl;    // 输入张量的元素的个数 BCHW
    for (const auto &x : input_dims.d ) {
        std::cout << x << " "; // 1 3 640 640
    }
    std::cout << std::endl;

    // outputName=output0, output_dims.nbDims=3
    std::cout << "outputName=" << outputName << ", output_dims.nbDims=" << output_dims.nbDims << std::endl;
    for (const auto &x : output_dims.d) {
        std::cout << x << " ";  // 1 84 8400 or 1x300x6
    }
    std::cout << std::endl;


    // assume the box outputs no more than 1000 boxes that conf >= nms; (1000)
    // left, top, right, bottom, confidence, class, keepflag(whether drop when NMS), 32 masks (7+32)
    // Initialize input buffers
    cpu_outpu_.resize(num_detections * detection_attribute_size);
    
    float* raw_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw_ptr), 3 * input_w * input_h * sizeof(float)));
    gpu_input_.reset(raw_ptr);

    raw_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw_ptr), detection_attribute_size * num_detections * sizeof(float)));
    gpu_output_.reset(raw_ptr);

    // gpu_input_.reset();
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(gpu_input_.get()), 3 * input_w * input_h * sizeof(float)));
    // Initialize output buffer
    // gpu_output_.reset();
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(gpu_output_.get()), detection_attribute_size * num_detections * sizeof(float)));
}

YOLOv11::~YOLOv11()
{

}

void YOLOv11::preprocess(cv::Mat& image) {
    // Preprocessing data on gpu
    preprocess_resize_gpu(image, gpu_input_.get(), input_h, input_w, *stream_);

    // int img_size = image.cols * image.rows * 3;
    // cudaMemcpyAsync(gpu_input_.get(), image.ptr(), img_size, cudaMemcpyHostToDevice, *stream_);
    // CUDA_CHECK(cudaStreamSynchronize(*stream_));

    // cuda_preprocess(gpu_input_.get(), image.cols, image.rows, gpu_input_.get(), input_w, input_h, *stream_);
    // CUDA_CHECK(cudaStreamSynchronize(*stream_));
}

void YOLOv11::infer(cv::Mat& image)
{
    preprocess(image);
    CUDA_CHECK(cudaStreamSynchronize(*stream_));

    // 绑定输入输出gpu显存
    context_->setTensorAddress(engine_->getIOTensorName(0), gpu_input_.get());
    context_->setTensorAddress(engine_->getIOTensorName(engine_->getNbIOTensors() - 1), gpu_output_.get());
    context_->enqueueV3(*stream_);
    CUDA_CHECK(cudaStreamSynchronize(*stream_));

    std::vector<Detection> objects;
    postprocess(objects);
    draw(image, objects);
}

void YOLOv11::postprocess(std::vector<Detection>& output)
{
    // Memcpy from device output buffer to host output buffer
    cudaMemcpyAsync(cpu_outpu_.data(), gpu_output_.get(), num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, *stream_);
    CUDA_CHECK(cudaStreamSynchronize(*stream_));

    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    // const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);
    for (int i=0; i<12; ++i) {
        std::cout << cpu_outpu_[i] << " ";
    }
    std::cout << std::endl;


    // for (int i = 0; i < det_output.cols; ++i) {
    //     // const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
    //     Point class_id_point;
    //     double score;
    //     minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

    //     if (score > conf_threshold) {
    //         const float cx = det_output.at<float>(0, i);
    //         const float cy = det_output.at<float>(1, i);
    //         const float ow = det_output.at<float>(2, i);
    //         const float oh = det_output.at<float>(3, i);
    //         Rect box;
    //         box.x = static_cast<int>((cx - 0.5 * ow));
    //         box.y = static_cast<int>((cy - 0.5 * oh));
    //         box.width = static_cast<int>(ow);
    //         box.height = static_cast<int>(oh);

    //         boxes.push_back(box);
    //         class_ids.push_back(class_id_point.y);
    //         confidences.push_back(score);
    //     }
    // }

    // vector<int> nms_result;
    // dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    // for (int i = 0; i < nms_result.size(); i++)
    // {
    //     Detection result;
    //     int idx = nms_result[i];
    //     result.class_id = class_ids[idx];
    //     result.conf = confidences[idx];
    //     result.bbox = boxes[idx];
    //     output.push_back(result);
    // }
}

void YOLOv11::draw(cv::Mat& image, const std::vector<Detection>& output)
{
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        if (ratio_h > ratio_w)
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else
        {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        rectangle(image, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), color, 3);

        // Detection box text
        std::string class_string = CLASS_NAMES[class_id] + ' ' + std::to_string(conf).substr(0, 4);
        cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, cv::FILLED);
        putText(image, class_string, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
}
