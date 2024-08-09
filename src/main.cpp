#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "PredictDetector.h"





int main() {
    PredictDetector predict_det("../models/det.onnx", "auto");
    std::unique_ptr<Ort::Session>& session = predict_det.GetSessionModel();
    // Your prediction code here
    
    auto inputNodeNames = predict_det.GetInputNames();

    std::cout << "Input names:" << std::endl;
    for (const auto& name : inputNodeNames) {
        std::cout << name << std::endl;
    }


    auto outputNodeNames = predict_det.GetOutputNames();
    std::cout << "Output names:" << std::endl;
    for (const auto& name : outputNodeNames) {
        std::cout << name << std::endl;
    }

    cv::Mat src  = cv::imread("../data/11.jpg");
    if (src.empty())
    {
        std::cout << "Could not load iamge ..." << std::endl;
        return -1;  
    }
    // predict_det.Predict(src);

    // cv::Mat src1 = predict_det.Preprocess(src);

    // std::vector<float> input_image = predict_det.Normalize(src1);

    // for (int i = 0; i < input_image.size(); i++) {  
    //     std::cout << input_image[i] << " ";
    // }

    auto ort_outputs = predict_det.Predict(src);
    // // 处理输出
        // // 打印输出张量
    for (size_t i = 0; i < ort_outputs.size(); ++i) {
        // 获取输出张量的形状信息
        Ort::TensorTypeAndShapeInfo shape_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = shape_info.GetShape();

        std::cout << "Output Tensor " << i << " Shape: ";
        for (size_t j = 0; j < output_shape.size(); ++j) {
            std::cout << output_shape[j] << " ";
        }
        std::cout << std::endl;

        // 获取输出张量的数据
        float* float_array = ort_outputs[i].GetTensorMutableData<float>();
        size_t total_elements = shape_info.GetElementCount();

        std::cout << "Output Tensor " << i << " Data: ";
        for (size_t j = 0; j < total_elements; ++j) {
            std::cout << float_array[j] << " ";
        }
        std::cout << std::endl;
    }

    predict_det.Postprocess(ort_outputs);
    cv::imshow("src", src);
    cv::waitKey(0);


    return 0;
}




// int main(int, char**) {
//     printf("Hello, from inspurocr!\n");

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
//     Ort::SessionOptions session_options;


    

//     printf("Available devices:\n");
//     printf("1. CPU\n");

//     cv::Mat src  = cv::imread("../data/lena.jpg");
//     if (src.empty())
//     {
//         std::cout << "Could not load iamge ..." << std::endl;
//         return -1;
//     }

//     cv::imshow("src", src);
//     cv::waitKey(0);




//     return 0;
// }
