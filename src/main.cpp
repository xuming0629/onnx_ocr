#include <iostream>
#include "PredictSystem.h"

int main()
{
    PredictSystem predictSystem;
    auto det_model = "../models/det/det.onnx";
    auto rec_model = "../models/rec/rec.onnx";
    auto rec_char_dict = "../models/rec_char_dict.txt";
    auto cls_model = "../models/cls/cls.onnx";
    predictSystem.LoadModel(det_model, rec_model, rec_char_dict, cls_model);

     // 读取输入图像
    cv::Mat img = cv::imread("../data/11.jpg");
    if (img.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    std::string results = predictSystem.ocr(img);
    std::cout << results << std::endl;


    return 0;
}