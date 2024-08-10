#include "PredictSystem.h"
#include "PredictDetector.h"
#include "PredictRecognizer.h"
#include "PredictClassifier.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

// 你可能需要包含这些依赖库的头文件
#include <onnxruntime_cxx_api.h>
#include "json.hpp"

using json = nlohmann::json;

static std::unique_ptr<PredictSystem> predictSystem;

extern "C" {
    void LoadModel(const char* det_model, const char* rec_model, const char* rec_char_dict, const char* cls_model) {
        predictSystem = std::make_unique<PredictSystem>();
        predictSystem->LoadModel(det_model, rec_model, rec_char_dict, cls_model);
    }

    const char* ocr(const char* image_path) {
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            return "{}";  // 如果图像加载失败，返回空的 JSON 对象
        }
        std::string result = predictSystem->ocr(img);
        // 将 std::string 转为 C 字符串并返回
        char* result_cstr = new char[result.length() + 1];
        std::strcpy(result_cstr, result.c_str());
        return result_cstr;
    }
}
