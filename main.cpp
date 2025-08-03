#include <iostream>
#include <string>
#include "yolov11.h"

int main()
{
    // init model
    YOLOv11 model("../Models/yolo11l_fp16.trt");

    if (true){
        model.input_image_path_ = "../asset/crossroad.jpg";
        cv::Mat img = cv::imread(model.input_image_path_);
        if (img.empty()) std::cerr << "Error reading image: " << std::endl;
        model.infer(img);
    }
    

    if (false) {
        std::string video_path = "../bubble.mp4"; 
        //path to video
        cv::VideoCapture cap(video_path);

        while (1)
        {
            cv::Mat image;
            cap >> image;

            if (image.empty()) break;

            // std::vector<Detection> objects;
            // model.preprocess(image);

            auto start = std::chrono::system_clock::now();
            model.infer(image);
            auto end = std::chrono::system_clock::now();
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);

        }

        // Release resources
        cv::destroyAllWindows();
        cap.release();
    }

    return 0;
}