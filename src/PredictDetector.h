#ifndef PREDICTDETECTOR_H
#define PREDICTDETECTOR_H

#include "PredictBase.h"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <array>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class PredictDetector : public PredictBase {
public:
    PredictDetector(const std::string& model_path, 
                    const std::string& device,
                    float db_thresh = 0.3f,
                    float box_thresh = 0.6f,
                    int max_candidates = 1000,
                    float unclip_ratio = 1.5f)
        : PredictBase(model_path, device),  // 调用基类构造函数
          db_thresh_(db_thresh),
          box_thresh_(box_thresh),
          max_candidates_(max_candidates),
          unclip_ratio_(unclip_ratio),
          src_height_(0),
          src_width_(0),
          short_size_(640)  // 默认短边大小
    {}

    ~PredictDetector() override = default;

    // Make these methods public so they can be accessed outside of the class
    std::unique_ptr<Ort::Session>& GetSessionModel() override {
        if (!session_model_) {
            session_model_ = std::make_unique<Ort::Session>(env, onnx_model.c_str(), session_options);
            std::cout << "Model loaded detector model successfully on " << device << std::endl;
        }
        return session_model_;
    }

    std::vector<std::string> GetInputNames() override {
        if (!session_model_) {
            throw std::runtime_error("Session model is not initialized.");
        }

        std::vector<std::string> input_names;
        size_t num_input_nodes = session_model_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; ++i) {
            auto input_name = session_model_->GetInputNameAllocated(i, allocator_);
            input_names.push_back(input_name.get());
        }
        return input_names;
    }

    std::vector<std::string> GetOutputNames() override {
        if (!session_model_) {
            throw std::runtime_error("Session model is not initialized.");
        }

        std::vector<std::string> output_names;
        size_t num_output_nodes = session_model_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; ++i) {
            auto output_name = session_model_->GetOutputNameAllocated(i, allocator_);
            output_names.push_back(output_name.get());
        }
        return output_names;
    }

    std::vector<std::vector<cv::Point2f>> Predict(cv::Mat& src_img);
    cv::Mat get_rotate_crop_image(const cv::Mat& frame, const std::vector<cv::Point2f>& vertices);

private:
    cv::Mat Preprocess(const cv::Mat& input_image);
    void Normalize(cv::Mat& img); 
    std::vector<std::vector<cv::Point2f>> Postprocess(std::vector<Ort::Value>& outputs);

    float contourScore(const cv::Mat& binary, const std::vector<cv::Point>& contour);
    void unclip(const std::vector<cv::Point2f>& inPoly, std::vector<cv::Point2f> &outPoly);
    

private:
    float db_thresh_;
    float box_thresh_;
    int max_candidates_;
    float unclip_ratio_;

    int src_height_;
    int src_width_;

    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_model_;
    int short_size_;
    std::vector<float> input_image_;
    cv::Mat dstimg_;
    std::vector<float> mean_values_ = {0.485f, 0.456f, 0.406f};
    std::vector<float> norm_values_ = {0.229f, 0.224f, 0.225f};
    
    const int longSideThresh = 3;
};

#endif // PREDICTDETECTOR_H
