#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "PredictRecognizer.h"

int main() {
    try {
        // Initialize the PredictDetector object with the model path and device
        PredictRecognizer predict_rec("../models/rec/rec.onnx", "auto");
        std::cout << "PredictRecongize initialized." << std::endl;

        // Ensure the ONNX Runtime session is initialized
        std::unique_ptr<Ort::Session>& session = predict_rec.GetSessionModel();
        std::cout << "ONNX Runtime session initialized." << std::endl;

        // Load the image
        cv::Mat src = cv::imread("textimg_6.jpg");
        if (src.empty()) {
            std::cerr << "Could not load image..." << std::endl;
            return -1;
        }
        std::cout << "Image loaded successfully." << std::endl;

        // Run the prediction
        predict_rec.Predict(src);
        // std::cout << "Prediction completed. Number of results: " << results.size() << std::endl;

        // // Process and display results
        // for (size_t i = 0; i < results.size(); i++) {
        //     cv::Mat textimg = predict_det.get_rotate_crop_image(src, results[i]);
        //     cv::imshow("Text Image", textimg);
        //     cv::waitKey(0);
        // }

        // // Clean up OpenCV windows
        // cv::destroyAllWindows();

    } catch (const std::exception& ex) {
        std::cerr << "An error occurred: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
