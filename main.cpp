#include <iostream>
#include <string>
#include "yolov11.h"

int main()
{
    // init model
    YOLOv11 model("../Models/yolo11l_fp16.trt");

    if (false){
        cv::Mat image = cv::imread("../asset/car.jpg");
        if (image.empty()) std::cerr << "Error reading image: " << std::endl;

        model.infer(image);
        cv::imshow("prediction", image);
        cv::waitKey(0);
    }
    

    else {
        std::string video_path = "../asset/frame.mp4"; 
        cv::VideoCapture cap(video_path);
        
        float total_dur = 0;
        while (true)
        {
            cv::Mat image;
            cap >> image;

            if (image.empty()) {
                std::cerr << "End of video or failed to read frame." << std::endl;
                break;
            }

            auto start = std::chrono::system_clock::now();
            model.infer(image);
            auto end = std::chrono::system_clock::now();
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            total_dur += tc;
            std::cout << tc << "ms" << std::endl;

            cv::imshow("prediction", image);
            cv::waitKey(1);
        }
        std::cout << total_dur << "ms" << std::endl;

        // Release resources
        cv::destroyAllWindows();
        cap.release();
    }

    return 0;
}