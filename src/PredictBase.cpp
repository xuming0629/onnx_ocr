#include "PredictBase.h"

/**
 * @brief Construct a new PredictBase object.
 * 
 * Initializes the ONNX model and selects the appropriate device for inference.
 * 
 * @param model_path Path to the ONNX model file.
 * @param device Target device for model inference ("cpu", "cuda", "auto").
 */
PredictBase::PredictBase(const std::string& model_path, const std::string& device)
    : onnx_model(model_path), device(device) {
    std::cout << "Initializing model on " << device << "..." << std::endl;
    session_options = GetSessionOptions();
    providers = SelectDevice(device);
}

/**
 * @brief Destroy the PredictBase object.
 * 
 * Releases resources used by the ONNX model and runtime session.
 */
PredictBase::~PredictBase() {
    std::cout << "Destroying model..." << std::endl;
}

/**
 * @brief Select the appropriate execution providers based on the target device.
 * 
 * @param device The target device ("cpu", "cuda", "auto").
 * @return std::vector<const char*> List of selected execution providers.
 */
std::vector<const char*> PredictBase::SelectDevice(const std::string& device) {
    if (device == "cuda" || (device == "auto" && IsCudaAvailable())) {
        std::cout << "Using CUDA..." << std::endl;
        return {"CUDAExecutionProvider", "CPUExecutionProvider"};
    }
    std::cout << "Using CPU..." << std::endl;
    return {"CPUExecutionProvider"};
}

/**
 * @brief Get the session options for the ONNX runtime.
 * 
 * Configures the session options including optimization level and threading.
 * 
 * @return Ort::SessionOptions Configured session options.
 */
Ort::SessionOptions PredictBase::GetSessionOptions() {
    Ort::SessionOptions sess_options;
    sess_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sess_options.SetIntraOpNumThreads(4);  // Configure this based on the system or application needs
    return sess_options;
}

/**
 * @brief Check if a CUDA execution provider is available.
 * 
 * @return true If CUDA is available.
 * @return false If CUDA is not available.
 */
bool PredictBase::IsCudaAvailable() {
    std::vector<std::string> available_providers = Ort::GetAvailableProviders();
    return std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider") != available_providers.end();
}
