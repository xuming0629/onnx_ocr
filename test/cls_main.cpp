#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "PredictClassifier.h"

int main() {
    try {
        // Initialize the PredictDetector object with the model path and device
        PredictClassifier predict_cls("../models/cls/cls.onnx", "auto");
        std::cout << "PredictRecongize initialized." << std::endl;

        // Ensure the ONNX Runtime session is initialized
        std::unique_ptr<Ort::Session>& session = predict_cls.GetSessionModel();
        std::cout << "ONNX Runtime session initialized." << std::endl;

        // Load the image
        cv::Mat src = cv::imread("textimg_9.jpg");
        if (src.empty()) {
            std::cerr << "Could not load image..." << std::endl;
            return -1;
        }
        std::cout << "Image loaded successfully." << std::endl;

        // Run the prediction
        int results = predict_cls.Predict(src);
        std::cout << "Prediction completed. Result: " << results << std::endl;
        // std::cout << "Prediction completed. Number of results: " << results.size() << std::endl;

        

    } catch (const std::exception& ex) {
        std::cerr << "An error occurred: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
