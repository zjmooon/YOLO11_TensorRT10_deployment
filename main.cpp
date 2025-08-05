#include <iostream>
#include <string>
#include "yolov11.h"

int main()
{
    // init model
    YOLOv11 model("../Models/yolo11l_fp16.trt");

    if (true){
        model.input_image_path_ = "../asset/car.jpg";
        cv::Mat image = cv::imread(model.input_image_path_);
        if (image.empty()) std::cerr << "Error reading image: " << std::endl;
        model.infer(image);

        cv::imshow("prediction", image);
        cv::waitKey(0);
    }
    

    if (false) {
        std::string video_path = "../asset/road1.mp4"; 
        cv::VideoCapture cap(video_path);

        while (1)
        {
            cv::Mat image;
            cap >> image;

            if (image.empty()) std::cerr << "Error reading image: " << std::endl;

            auto start = std::chrono::system_clock::now();
            model.infer(image);
            auto end = std::chrono::system_clock::now();
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            std::cout << tc << "ms" << std::endl;

            cv::imshow("prediction", image);
            cv::waitKey(1);
        }

        // Release resources
        cv::destroyAllWindows();
        cap.release();
    }

    return 0;
}