#ifndef PREDICTDETECTOR_H
#define PREDICTDETECTOR_H

#include "PredictBase.h"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>

class PredictDetector : public PredictBase {
public:
    // PrediceDetctor类构造函数
    PredictDetector(const std::string& model_path, 
                    const std::string& device,
                    float db_thres = 0.3f,
                    float box_thres = 0.6f,
                    int max_candidates = 1000,
                    float unclip_ratio = 1.5f)
        : PredictBase(model_path, device),  // 调用基类构造函数
          db_thres(db_thres),
          box_thres(box_thres),
          max_candidates(max_candidates),
          unclip_ratio(unclip_ratio) {
          }
    // PrediceDetctor类析构函数
    ~PredictDetector() override = default;


    std::unique_ptr<Ort::Session>& GetSessionModel() override {
        if (!session_model) {
            session_model = std::make_unique<Ort::Session>(env, onnx_model.c_str(), session_options);
            std::cout << "Model loaded detector model successfully on " << device << std::endl;
        }
        return session_model;
    }

    std::vector<std::string> GetInputNames() {
        if (!session_model) {
            throw std::runtime_error("Session model is not initialized.");
        }

        std::vector<std::string> input_names;
        size_t numInputNodes = session_model->GetInputCount();
        for (size_t i = 0; i < numInputNodes; ++i) {
            auto inputName = session_model->GetInputNameAllocated(i, allocator);
            input_names.push_back(inputName.get()); // Store as std::string
        }
        return input_names;
    }

    std::vector<std::string> GetOutputNames() {
        if (!session_model) {
            throw std::runtime_error("Session model is not initialized.");
        }

        std::vector<std::string> output_names;
        size_t numOutputNodes = session_model->GetOutputCount();
        for (size_t i = 0; i < numOutputNodes; ++i) {
            auto outputName = session_model->GetOutputNameAllocated(i, allocator);
            output_names.push_back(outputName.get()); // Store as std::string
        }
        return output_names;
    }

    void Predict(const cv::Mat& image, std::vector<float>& output) {
        
    }

private:
    // 新增的成员变量
    float db_thres;
    float box_thres;
    int max_candidates;
    float unclip_ratio;

    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session_model;
};

#endif // PREDICTDETECTOR_H
