#include "PredictSystem.h"
#include <iostream>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "PredictDetector.h"
#include "PredictRecognizer.h"
#include "PredictClassifier.h"
#include "json.hpp"

using json = nlohmann::json;

class PredictSystem::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    void LoadModel(const std::string& det_model, const std::string& rec_model, const std::string& rec_char_dict, const std::string& cls_model) {
        try {
            detector_ = std::make_unique<PredictDetector>(det_model, "auto");
            recognizer_ = std::make_unique<PredictRecognizer>(rec_model, "auto", rec_char_dict);
            classifier_ = std::make_unique<PredictClassifier>(cls_model, "auto");
            std::cout << "Models loaded successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading models: " << e.what() << std::endl;
        }
    }

    std::string ocr(cv::Mat img) {
        if (img.empty()) {
            std::cerr << "Image is empty." << std::endl;
            return "{}";
        }
        if (!detector_ || !recognizer_ || !classifier_) {
            std::cerr << "Models are not loaded properly." << std::endl;
            return "{}";
        }

        json output_json = json::array();
        std::vector<std::vector<cv::Point2f>> detected_boxes = detector_->Predict(img);

        for (size_t i = 0; i < detected_boxes.size(); ++i) {
            const auto& box = detected_boxes[i];
            cv::Mat textimg = detector_->get_rotate_crop_image(img, box);
            int angle = classifier_->Predict(textimg);
            if (angle == 180) {
                cv::rotate(textimg, textimg, 1);
            }
            std::string text = recognizer_->Predict(textimg);
            json result_json;
            result_json["text"] = text;

            json box_json = json::array();
            for (const auto& point : box) {
                box_json.push_back({{"x", point.x}, {"y", point.y}});
            }
            result_json["box"] = box_json;
            output_json.push_back(result_json);
        }
        return output_json.dump(4);
    }

private:
    std::unique_ptr<PredictDetector> detector_;
    std::unique_ptr<PredictRecognizer> recognizer_;
    std::unique_ptr<PredictClassifier> classifier_;
};

PredictSystem::PredictSystem() : pImpl(std::make_unique<Impl>()) {}

PredictSystem::~PredictSystem() = default;

void PredictSystem::LoadModel(const std::string& det_model, const std::string& rec_model, const std::string& rec_char_dict, const std::string& cls_model) {
    pImpl->LoadModel(det_model, rec_model, rec_char_dict, cls_model);
}

std::string PredictSystem::ocr(cv::Mat img) {
    return pImpl->ocr(std::move(img));
}
