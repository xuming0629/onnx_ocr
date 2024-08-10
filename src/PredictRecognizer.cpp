#include "PredictRecognizer.h"

PredictRecognizer::PredictRecognizer(const std::string& model_path, const std::string& device, const std::string& rec_char_dic)
    : PredictBase(model_path, device), width_(320), height_(48) {
    session_model_ = std::move(GetSessionModel());
    LoadAlphabet(rec_char_dic);

}

std::unique_ptr<Ort::Session>& PredictRecognizer::GetSessionModel() {
    if (!session_model_) {
        session_model_ = std::make_unique<Ort::Session>(env, onnx_model.c_str(), session_options);
        std::cout << "Model loaded recognizer model successfully on " << device << std::endl;
    }
    return session_model_;
}

void PredictRecognizer::LoadAlphabet(const std::string& filename) {
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

std::vector<std::string> PredictRecognizer::GetInputNames() {
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

std::vector<std::string> PredictRecognizer::GetOutputNames() {
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

cv::Mat PredictRecognizer::Preprocess(const cv::Mat& image) {
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

    return output_img;
}

void PredictRecognizer::Normalize(cv::Mat& img) {
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

std::string PredictRecognizer::Predict(const cv::Mat& image) {
    if (!session_model_) {
        throw std::runtime_error("Session model is not initialized.");
    }

    cv::Mat preprocessed_image = Preprocess(image);
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

    // std::cout << "Output size: " << ort_outputs.size() << std::endl;
    
    return Postprocess(ort_outputs);
}

std::string PredictRecognizer::Postprocess(std::vector<Ort::Value>& outputs) {
   
    // std::cout << "start rec Postprocess" << std::endl;
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
