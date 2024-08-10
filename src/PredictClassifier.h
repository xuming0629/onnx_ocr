#ifndef PREDICTCLASSIFY_H
#define PREDICTCLASSIFY_H

#include "PredictBase.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <array>
#include <vector>

class PredictClassifier : public PredictBase {
public:
    PredictClassifier(const std::string& model_path, const std::string& device) 
        : PredictBase(model_path, device) {
            session_model_ = std::move(GetSessionModel());
        }

    std::unique_ptr<Ort::Session>& GetSessionModel() override {
        if (!session_model_) {
            session_model_ = std::make_unique<Ort::Session>(env, onnx_model.c_str(), session_options);
            std::cout << "Model loaded successfully on " << device << std::endl;
        }
        return session_model_;
    }

    std::vector<std::string> GetInputNames() {
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

    std::vector<std::string> GetOutputNames() {
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

    std::vector<std::vector<int64_t>> GetOutputShapes() {
        if (!session_model_) {
            throw std::runtime_error("Session model is not initialized.");
        }

        std::vector<std::vector<int64_t>> output_shapes;
        size_t num_output_nodes = session_model_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; ++i) {
		    Ort::TypeInfo output_type_info = session_model_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            auto output_dims = output_tensor_info.GetShape();
		    output_shapes.push_back(output_dims);
        }
        return output_shapes;
    }

    cv::Mat Preprocess(const cv::Mat& image) {
         cv::Mat dstimg;
        int target_width = 320;  // 目标宽度
        int target_height = 48;  // 目标高度

        int h = image.rows;
        int w = image.cols;

        // 计算缩放比例，保持宽高比
        float scale = std::min(target_width / static_cast<float>(w), target_height / static_cast<float>(h));

        // 根据比例缩放图像
        int resized_w = static_cast<int>(w * scale);
        int resized_h = static_cast<int>(h * scale);

        cv::resize(image, dstimg, cv::Size(resized_w, resized_h), cv::INTER_LINEAR);

        // 创建一个目标尺寸的图像，用白色（或黑色）填充
        cv::Mat output_img(target_height, target_width, dstimg.type(), cv::Scalar(255, 255, 255)); // 白色填充背景
        // cv::Scalar(0, 0, 0) 用于黑色填充背景

        // 将缩放后的图像放在中心
        dstimg.copyTo(output_img(cv::Rect((target_width - resized_w) / 2, (target_height - resized_h) / 2, resized_w, resized_h)));

        std::cout << "resized_w: " << resized_w << ", resized_h: " << resized_h << std::endl;
        std::cout << "Output image size: " << output_img.size() << std::endl;

        return output_img;
    }

    void Normalize(cv::Mat& img) {
        int row = img.rows;
        int col = img.cols;
        this->input_image_.resize(this->height_ * this->width_ * img.channels());
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < width_; j++) {
                    if (j < col) {
                        float pix = img.ptr<uchar>(i)[j * 3 + c];
                        this->input_image_[c * row * width_ + i * width_ + j] = (pix / 255.0 - 0.5) / 0.5;
                    } else {
                        this->input_image_[c * row * width_ + i * width_ + j] = 0;
                    }
                }
            }
        }
    }

    int Predict(cv::Mat& img) {
        cv::Mat dst_img = Preprocess(img);
        Normalize(dst_img);
        if (input_image_.size() != 3 * this->height_ * this->width_) {
            throw std::runtime_error("Input image size does not match expected size.");
        }
        
        std::array<int64_t, 4> input_shape_{1, 3, this->height_, this->width_};
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

        auto input_names = GetInputNames();
        auto output_names = GetOutputNames();

        std::vector<const char*> input_names_cstr;
        for (const auto& name : input_names) {
            input_names_cstr.push_back(name.c_str());
        }

        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names) {
            output_names_cstr.push_back(name.c_str());
        }

        std::vector<Ort::Value> ort_outputs = session_model_->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), &input_tensor_, 1, output_names_cstr.data(), output_names_cstr.size());
        std::cout << "Output size: " << ort_outputs.size() << std::endl;

        int tagIdx = Postprocess(ort_outputs);
        int angle = label_list[tagIdx];
        return angle;
    }

    int Postprocess(std::vector<Ort::Value>& outputs) {
        const float* pdata = outputs[0].GetTensorMutableData<float>();
        auto num_out = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        int max_id = 0;
	    float max_prob = -1;
	    for (int i = 0; i < num_out; i++) {
		    if (pdata[i] > max_prob) {
			    max_prob = pdata[i];
			    max_id = i;
		    }   
	    }
        int label_list[2] = { 0, 180 };

	    return max_id;
    }



private:
    const int label_list[2] = { 0, 180 };
    int width_ = 192;
    int height_ = 48;
    int num_out;
    std::vector<float> input_image_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_model_;
};

#endif // PREDICTCLASSIFY_H
