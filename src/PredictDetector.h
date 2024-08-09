#ifndef PREDICTDETECTOR_H
#define PREDICTDETECTOR_H

#include "PredictBase.h"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <array> 

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

    std::vector<Ort::Value> Predict(const cv::Mat& src_img) {
        if (!session_model) {
            throw std::runtime_error("Session model is not initialized.");
        }
        // TODO: Implement the prediction logic for the detector
        int h = src_img.rows;
        int w = src_img.cols;
        dstimg = this->Preprocess(src_img);
        // TODO: Call the inference function of the predictor
        this->Normalize(dstimg);
        // for (int i = 0; i < input_image.size(); i++) {  
        //     std::cout << input_image[i] << " ";
        // }

        std::array<int64_t, 4> input_shape{ 1, 3, dstimg.rows, dstimg.cols };

        for (int i=0; i<input_shape.size(); i++) {
            std::cout << input_shape[i] << " ";
        }

        std::cout << std::endl;

        // 创建输入张量
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());

        // 获取输入和输出名称
        auto input_names = GetInputNames();
        auto output_names = GetOutputNames();

        // 将输入和输出名称转换为 const char* const*
        std::vector<const char*> input_names_cstr;
        for (const auto& name : input_names) {
            input_names_cstr.push_back(name.c_str());
        }

        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names) {
            output_names_cstr.push_back(name.c_str());
        }

        // 执行推理
        std::vector<Ort::Value> ort_outputs = session_model->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), &input_tensor, 1, output_names_cstr.data(), output_names_cstr.size());        
        return ort_outputs;

    }


    void Postprocess(const std::vector<Ort::Value>& outputs) {
        const float* floatArray = ort_outputs[0].GetTensorMutableData<float>();
        int outputCount = 1;
        for(int i=0; i < ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().size(); i++)
        {
            int dim = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(i);
            outputCount *= dim;
        }
        cv::Mat binary(dstimg.rows, dstimg.cols, CV_32FC1);
        
    }

public:
    cv::Mat Preprocess(const cv::Mat& input_image) {
        if (input_image.empty()) {
            throw std::runtime_error("Input image is empty.");  
        }
        // 将 BGR 图像转换为 RGB
        cv::Mat rgb_image;
        cv::cvtColor(input_image, rgb_image, cv::COLOR_BGR2RGB);

        // 获取图像的高度和宽度
        int original_height = input_image.rows;
        int original_width = input_image.cols;

        // 初始化缩放比例
        float scale_height = 1.0f;
        float scale_width = 1.0f;

        // 根据长宽比调整缩放比例
        if (original_height < original_width) {
            scale_height = static_cast<float>(short_size) / static_cast<float>(original_height);
            float target_width = static_cast<float>(original_width) * scale_height;
            target_width = target_width - static_cast<int>(target_width) % 32;
            target_width = std::max(32.0f, target_width);
            scale_width = target_width / static_cast<float>(original_width);
        } else {
            scale_width = static_cast<float>(short_size) / static_cast<float>(original_width);
            float target_height = static_cast<float>(original_height) * scale_width;
            target_height = target_height - static_cast<int>(target_height) % 32;
            target_height = std::max(32.0f, target_height);
            scale_height = target_height / static_cast<float>(original_height);
        }

        // 调整图像大小
        cv::Mat resized_image;
        cv::resize(rgb_image, resized_image, cv::Size(static_cast<int>(scale_width * original_width), static_cast<int>(scale_height * original_height)), 0, 0, cv::INTER_LINEAR);

        return resized_image;
    }

    void Normalize(cv::Mat& img) {
        int rows = img.rows;
        int cols = img.cols;
        int channels = img.channels();

        input_image.resize(rows * cols * channels);

        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    float pixel = img.ptr<uchar>(i)[j * channels + c];
                    input_image[c * rows * cols + i * cols + j] = (pixel / 255.0f - mean_values[c]) / norm_values[c];
                }
            }
        }
    }



private:
    // 新增的成员变量
    float db_thres;
    float box_thres;
    int max_candidates;
    float unclip_ratio;

    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session_model;
    int short_size = 640;  // 假设模型输入要求短边的大小
    std::vector<float> input_image;
    cv::Mat dstimg;
    std::vector<float> mean_values = {0.485f, 0.456f, 0.406f};  // 通常用于图像标准化的均值
    std::vector<float> norm_values = {0.229f, 0.224f, 0.225f};  // 通常用于图像标准化的标准差
};

#endif // PREDICTDETECTOR_H
