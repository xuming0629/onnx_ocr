#ifndef PREDICTRECOGNIZER_H
#define PREDICTRECOGNIZER_H

#include "PredictBase.h"
#include <opencv2/opencv.hpp>  // Include OpenCV header for cv::Mat
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class PredictRecognizer : public PredictBase {
public:
    PredictRecognizer(const std::string& model_path, const std::string& device, const std::string& rec_char_dic);

    std::unique_ptr<Ort::Session>& GetSessionModel() override;

    void LoadAlphabet(const std::string& filename);

    std::vector<std::string> GetInputNames() override;

    std::vector<std::string> GetOutputNames() override;

    cv::Mat Preprocess(const cv::Mat& image);

    void Normalize(cv::Mat& img);

    std::string Predict(const cv::Mat& image);

    std::string Postprocess(std::vector<Ort::Value>& outputs);

private:
    int width_;
    int height_;
    std::vector<float> input_image_;
    std::vector<std::string> alphabet_;
    std::vector<int> preb_label_;
    int names_len;

    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_model_;
};

#endif // PREDICTRECOGNIZER_H
