#include <iostream>
#include <string>
#include "yolov11.h"


/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main()
{ 
    std::string video_path = "../frame.mp4"; 
    // init model
    YOLOv11 model("../Models/yolo11n.trt", logger);

    if (true) {
        //path to video
        cv::VideoCapture cap(video_path);

        while (1)
        {
            cv::Mat image;
            cap >> image;

            if (image.empty()) break;

            std::vector<Detection> objects;
            model.preprocess(image);

            auto start = std::chrono::system_clock::now();
            model.infer();
            auto end = std::chrono::system_clock::now();

            model.postprocess(objects);
            // model.draw(image, objects);

            // auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            // printf("cost %2.4lf ms\n", tc);

            // imshow("prediction", image);
            // waitKey(1);
        }

        // Release resources
        cv::destroyAllWindows();
        cap.release();
    }
    // else {
    //     // path to folder saves images
    //     for (const auto& imagePath : imagePathList)
    //     {
    //         // open image
    //         Mat image = imread(imagePath);
    //         if (image.empty())
    //         {
    //             cerr << "Error reading image: " << imagePath << endl;
    //             continue;
    //         }

    //         vector<Detection> objects;
    //         model.preprocess(image);

    //         auto start = std::chrono::system_clock::now();
    //         model.infer();
    //         auto end = std::chrono::system_clock::now();

    //         model.postprocess(objects);
    //         model.draw(image, objects);

    //         auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    //         printf("cost %2.4lf ms\n", tc);

    //         model.draw(image, objects);
    //         imshow("Result", image);

    //         waitKey(0);
    //     }
    // }

    return 0;
}