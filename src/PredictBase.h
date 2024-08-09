#ifndef PREDICTBASE_H
#define PREDICTBASE_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

class PredictBase {
public:
    PredictBase(const std::string& model_path, const std::string& device)
        : onnx_model(model_path), device(device) {
        std::cout << "Initializing model on " << device << "..." << std::endl;
        session_options = GetSessionOptions();
        providers = SelectDevice(device);
    }

    virtual ~PredictBase() {}

    // 纯虚函数，子类必须实现
    virtual std::unique_ptr<Ort::Session>& GetSessionModel() = 0;
    virtual std::vector<std::string> GetInputNames() = 0;
    virtual std::vector<std::string> GetOutputNames() = 0;
    // virtual std::vector<int64_t> GetInputDimensions(const std::string& input_name) = 0;
    //virtual std::vector<int64_t> GetOutputDimensions(const std::string& output_name) = 0;

protected:
    std::string onnx_model;
    std::string device;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"};
    Ort::SessionOptions session_options;
    std::vector<const char*> providers;
   

    std::unique_ptr<Ort::Session> session_model;

    std::vector<const char*> SelectDevice(const std::string& device) {
        if (device == "cuda" || (device == "auto" && IsCudaAvailable())) {
            std::cout << "Using CUDA..." << std::endl;
            return {"CUDAExecutionProvider", "CPUExecutionProvider"};
        }
        std::cout << "Using CPU..." << std::endl;
        return {"CPUExecutionProvider"};
    }

    Ort::SessionOptions GetSessionOptions() {
        Ort::SessionOptions sess_options;
        sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        sess_options.SetIntraOpNumThreads(4);
        return sess_options;
    }

    bool IsCudaAvailable() {
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        return std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") != available_providers.end();
    }

};

#endif // PREDICTBASE_H
