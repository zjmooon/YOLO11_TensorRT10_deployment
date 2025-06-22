#include "yolov11.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>


static Logger logger;


YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger)
{
    init(model_path, logger);
}


void YOLOv11::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    // Read the engine file
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Get input and output sizes of the model
    const char* inputName = engine->getIOTensorName(0);
    int dimNums = engine->getNbIOTensors();
    const char* outputName = engine->getIOTensorName(dimNums - 1);
    Dims input_dims = engine->getTensorShape(inputName);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    Dims output_dims = engine->getTensorShape(outputName); // 1 300 6
    num_detections = output_dims.d[1];
    detection_attribute_size = output_dims.d[2];

    std::cout << engine->getNbIOTensors() << std::endl;   // 2, 打印模型所有dims包括输入输出

    // inputName=images, input_dims.nbDims=4
    std::cout << "inputName=" << inputName << ", input_dims.nbDims=" << input_dims.nbDims << std::endl;    // 4, 输入张量的元素的个数 NCHW
    for (const auto &x : input_dims.d ) {
        std::cout << x << " "; // 1 3 640 640 0 0 0 0 
    }
    std::cout << std::endl;

    // outputName=output0, output_dims.nbDims=3
    std::cout << "outputName=" << outputName << ", output_dims.nbDims=" << output_dims.nbDims << std::endl;
    for (const auto &x : output_dims.d ) {
        std::cout << x << " ";  // 1 300 6 0 0 0 0 0 (batch，检测框最大数量，属性（x1,y1,x2,y2,conf,class）)
    }
    std::cout << std::endl;


    // Initialize input buffers
    cpu_output_buffer = new float[detection_attribute_size * num_detections];
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    // Initialize output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

    cuda_preprocess_init(MAX_IMAGE_SIZE);

    CUDA_CHECK(cudaStreamCreate(&stream));

    // 绑定输入输出gpu显存
    context->setTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
    context->setTensorAddress(engine->getIOTensorName(engine->getNbIOTensors() - 1), gpu_buffers[1]);


}

YOLOv11::~YOLOv11()
{
    // Release stream and buffers
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    delete[] cpu_output_buffer;

    // Destroy the engine
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;
}

void YOLOv11::preprocess(Mat& image) {
    // Preprocessing data on gpu
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11::infer()
{
//     context->enqueueV2((void**)gpu_buffers, stream, nullptr);

    this->context->enqueueV3(this->stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11::postprocess(vector<Detection>& output)
{
    // Memcpy from device output buffer to host output buffer
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    // const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);
    for (int i=0; i<12; ++i) {
        std::cout << cpu_output_buffer[i] << " ";
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

void YOLOv11::draw(Mat& image, const vector<Detection>& output)
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

        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Detection box text
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, FILLED);
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}