#ifndef PREDICTRECOGNIZER_H
#define PREDICTRECOGNIZER_H

#include "PredictBase.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class PredictRecognizer : public PredictBase {
public:
    PredictRecognizer(const std::string& model_path, const std::string& device, const std::string& rec_char_dic) 
        : PredictBase(model_path, device), width_(320), height_(48) {
        LoadAlphabet(rec_char_dic);
    }

    std::unique_ptr<Ort::Session>& GetSessionModel() override {
        if (!session_model_) {
            session_model_ = std::make_unique<Ort::Session>(env, onnx_model.c_str(), session_options);
            std::cout << "Model loaded recognize model successfully on " << device << std::endl;
        }
        return session_model_;
    }

    void LoadAlphabet(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        std::string line;
        alphabet_.clear();

        while (std::getline(ifs, line)) {
            alphabet_.push_back(line);
        }

        alphabet_.push_back(" ");
        names_len = alphabet_.size();
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

    cv::Mat Preprocess(const cv::Mat& image) {
        cv::Mat dstimg;
        int h = image.rows;
        int w = image.cols;

        float ratio = w / static_cast<float>(h);
        int resized_w = static_cast<int>(std::ceil(this->height_ * ratio));

        if (resized_w > this->width_) {
            resized_w = this->width_;
        }

        cv::resize(image, dstimg, cv::Size(resized_w, this->height_), 0, 0, cv::INTER_LINEAR);
        return dstimg;
    }

    void Normalize(cv::Mat& img) {
        int row = img.rows;
        int col = img.cols;
        int channels = img.channels();
        int width_ = col;

        input_image_.resize(row * width_ * channels);

        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < width_; j++) {
                    if (j < col) {
                        float pix = img.ptr<uchar>(i)[j * channels + c];
                        input_image_[c * row * width_ + i * width_ + j] = (pix / 255.0f - 0.5f) / 0.5f;
                    } else {
                        input_image_[c * row * width_ + i * width_ + j] = 0.0f;
                    }
                }
            }
        }
    }
    
    std::string Predict(const cv::Mat& image) {
        if (!session_model_) {
            throw std::runtime_error("Session model is not initialized.");
        }

        cv::Mat preprocessed_image = Preprocess(image);
        std::cout << preprocessed_image.size() << std::endl;
        Normalize(preprocessed_image);

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
        
        return Postprocess(ort_outputs);
    }

    std::string Postprocess(std::vector<Ort::Value>& outputs) {
        std::cout << "start rec Postprocess" << std::endl;
        const float* pdata = outputs[0].GetTensorMutableData<float>();
        int h = outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(2);
        int w = outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(1);
        preb_label_.resize(w);

        for (int i = 0; i < w; i++) {
            int one_label_idx = 0;
            float max_data = -10000;
            for (int j = 0; j < h; j++) {
                float data_ = pdata[i*h + j];
                if (data_ > max_data) {
                    max_data = data_;
                    one_label_idx = j;
                }
            }
            preb_label_[i] = one_label_idx;
        }

        std::vector<int> no_repeat_blank_label;
        for (size_t elementIndex = 0; elementIndex < w; ++elementIndex) {
            if (preb_label_[elementIndex] != 0 && !(elementIndex > 0 && preb_label_[elementIndex - 1] == preb_label_[elementIndex])) {
                no_repeat_blank_label.push_back(preb_label_[elementIndex] - 1);
            }
        }

        std::string plate_text;
        for (int i = 0; i < static_cast<int>(no_repeat_blank_label.size()); i++) {
            plate_text += alphabet_[no_repeat_blank_label[i]];
        }

        return plate_text;
    }

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
