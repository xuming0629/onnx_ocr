#ifndef PREDICTRECOGNIZER_H
#define PREDICTRECOGNIZER_H

#include "PredictBase.h"

class PredictRecognizer : public PredictBase {
public:
    PredictRecognizer(const std::string& model_path, const std::string& device) 
        : PredictBase(model_path, device) {}



            std::unique_ptr<Ort::Session>& GetSessionModel() override {
        if (!session_model_) {
            session_model_ = std::make_unique<Ort::Session>(env, onnx_model.c_str(), session_options);
            std::cout << "Model loaded recongnize model successfully on " << device << std::endl;
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

    cv::Mat Preprocess(const cv::Mat& image) {
    cv::Mat dstimg;
    int h = image.rows;  // Image height
    int w = image.cols;  // Image width

    const float ratio = w / static_cast<float>(h);  // Calculate aspect ratio
    int resized_w = static_cast<int>(std::ceil(this->height_ * ratio));  // Calculate width based on aspect ratio

    // Check if the resized width exceeds the maximum allowed width
    if (resized_w > this->width_) {
        resized_w = this->width_;  // Set to maximum width if exceeded
    }

    // Resize the image while maintaining the aspect ratio
    cv::resize(image, dstimg, cv::Size(resized_w, this->height_), 0, 0, cv::INTER_LINEAR);

    return dstimg;
}

    void Normalize(cv::Mat& img) {
        int row = img.rows;
        int col = img.cols;
        int channels = img.channels();
        int width_ = col; // Assuming inpWidth refers to the width of the image

        // Resize the input_image_ to hold the normalized data
        input_image_.resize(row * width_ * channels);

        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < width_; j++) {
                    if (j < col) {
                        float pix = img.ptr<uchar>(i)[j * channels + c];
                        // Normalize the pixel value to [0, 1], then scale and shift to [-1, 1]
                        this->input_image_[c * row * width_ + i * width_ + j] = (pix / 255.0f - 0.5f) / 0.5f;
                    } else {
                        // If the column exceeds the actual width, pad with zeros
                        this->input_image_[c * row * width_ + i * width_ + j] = 0.0f;
                    }
                }
            }
        }
    }
    
    void Predict(const cv::Mat& image) {
        if (!session_model_) {
            throw std::runtime_error("Session model is not initialized.");
        }

        cv::Mat preprocessed_image = Preprocess(image);
        std::cout << preprocessed_image.size() << std::endl;
        Normalize(preprocessed_image);

        // Ensure input_image_ has the correct size
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

        // Run inference
        std::vector<Ort::Value> ort_outputs = session_model_->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), &input_tensor_, 1, output_names_cstr.data(), output_names_cstr.size());

        std::cout << "Output size: " << ort_outputs.size() << std::endl;
    }
private:
    int width_ = 320;
    int height_ = 48;
    std::vector<float> input_image_;
    std::vector<std::string> alphabet;


    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_model_;
    



};

#endif // PREDICTRECOGNIZER_H
