#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "PredictDetector.h"





int main() {
    PredictDetector predict_det("../models/en_det_model.onnx", "auto");
    std::unique_ptr<Ort::Session>& session = predict_det.GetSessionModel();
    // Your prediction code here

    auto inputNodesNum = session->GetInputCount();
    Ort::AllocatorWithDefaultOptions allocator;
	auto temp_input_name0 = session->GetInputNameAllocated(0, allocator);
	
    std::vector<const char*> inputNodeNames; //
	std::vector<const char*> outputNodeNames;//
	std::vector<int64_t> inputTensorShape; //
    inputNodeNames.push_back(temp_input_name0.get());
    std::cout << "Input names:" << std::endl;
    for (const auto& name : inputNodeNames) {
        std::cout << name << std::endl;
    }
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
