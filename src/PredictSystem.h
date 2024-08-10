#include <iostream>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "PredictDetector.h"
#include "PredictRecognizer.h"
#include "PredictClassifier.h"

#include "json.hpp"

using json = nlohmann::json;

class PredictSystem {
public:
    PredictSystem() = default;
    ~PredictSystem() = default;

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
        // Implementation of OCR prediction logic using detector_, recognizer_, and classifier_
        
        // 检查输入图像是否为空
        if (img.empty()) {
            std::cerr << "Image is empty." << std::endl;
            return "{}";  // 返回空的 JSON 对象
        }
        // 检查模型是否已加载
        if (!detector_ || !recognizer_ || !classifier_) {
            std::cerr << "Models are not loaded properly." << std::endl;
            return "{}";
        }

        json output_json = json::array();

        // 1. 检测文本区域
        std::vector<std::vector<cv::Point2f>> detected_boxes = detector_->Predict(img);

        for (size_t i = 0; i < detected_boxes.size(); ++i) {
            const auto& box = detected_boxes[i];

            // 裁剪检测到的文本区域
            cv::Mat textimg = detector_->get_rotate_crop_image(img, box);
            int angle = classifier_->Predict(textimg);
            if (angle == 180) {
                cv::rotate(textimg, textimg, 1);
            }
            // 识别文本
            std::string text = recognizer_->Predict(textimg);
            std::cout << "Recognized text: " << text << std::endl;

            // 构建单个结果的 JSON 对象
            json result_json;
            result_json["text"] = text;

            json box_json = json::array();
            for (const auto& point : box) {
                box_json.push_back({{"x", point.x}, {"y", point.y}});
            }
            result_json["box"] = box_json;

            // 将该结果添加到输出数组
            output_json.push_back(result_json);
        }
        return output_json.dump(4);  // 4表示缩进级别，格式化输出
    }

private:
    std::unique_ptr<PredictDetector> detector_;
    std::unique_ptr<PredictRecognizer> recognizer_;
    std::unique_ptr<PredictClassifier> classifier_;
};
